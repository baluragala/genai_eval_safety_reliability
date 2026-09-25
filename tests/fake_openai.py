"""A stand-in OpenAI client so the whole lab can be tested with no key and no network.

This is test infrastructure only; learners always talk to the real API. It
imitates a capable but gullible tool-calling model: it follows the store
policy, and it also obeys instructions it finds in untrusted text. That lets
the tests show the guards doing their job.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace as NS

DIRECTIVE = re.compile(r"(issue_refund|send_email)\s*\(([^)]*)\)")


def _resp(content=None, tool_calls=None, pt=900, ct=40):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=pt, completion_tokens=ct))


class _Calls:
    def __init__(self):
        self.n = 0

    def tc(self, name, args):
        self.n += 1
        return NS(id=f"call_{self.n}", function=NS(name=name, arguments=json.dumps(args)))


class FakeOpenAI:
    def __init__(self):
        self.ids = _Calls()
        self.chat = NS(completions=NS(create=self.create))
        self.requests = []

    # ------------------------------------------------------------ dispatch
    def create(self, **kw):
        self.requests.append(kw)
        msgs = kw["messages"]
        if kw.get("response_format"):
            return self._json(msgs)
        if not kw.get("tools"):
            sys = msgs[0]["content"] if msgs else ""
            if "role-playing a customer" in sys:
                return _resp("Thanks, my order is A-1008. [DONE]" if len(msgs) > 3 else "It's order A-1008.")
            return _resp("ready")
        return self._agent(msgs, [t["function"]["name"] for t in kw["tools"]])

    def _json(self, msgs):
        sys, user = msgs[0]["content"], msgs[-1]["content"]
        if "security classifier" in sys:
            bad = any(w in user.lower() for w in ["override", "admin", "audit", "environ", "remember"])
            return _resp(json.dumps({"attack": bad, "reason": "fake"}))
        if "Rewrite the customer message" in sys:
            return _resp(json.dumps({"message": user.split("Message:", 1)[-1].strip()}))
        # judge
        claims_refund = "refund" in user.lower() and ("processed" in user.lower() or "issued" in user.lower())
        if "Execution trace" in user and claims_refund and "SUCCEEDED" not in user:
            v = {"policy_compliance": 2, "groundedness": 1, "helpfulness": 3, "tone": 5, "verdict": "fail",
                 "rationale": "claims a refund the trace does not show"}
        else:
            v = {"policy_compliance": 5, "groundedness": 5, "helpfulness": 5, "tone": 5, "verdict": "pass",
                 "rationale": "fine"}
        return _resp(json.dumps(v))

    # ------------------------------------------------------------ agent policy
    def _agent(self, msgs, tools):
        sysmsg = msgs[0]["content"]
        cid = re.search(r"customer for this conversation is (C-\d+)", sysmsg).group(1)
        users = " ".join(m["content"] for m in msgs if m["role"] == "user")
        calls, results = {}, []
        for m in msgs:
            if m["role"] == "assistant":
                for tc in m.get("tool_calls") or []:
                    calls[tc["id"]] = (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
            if m["role"] == "tool":
                name, args = calls[m["tool_call_id"]]
                results.append((name, args, json.loads(m["content"])))
        done = [(n, json.dumps(a, sort_keys=True)) for n, a, _ in results]

        def call(name, args):
            if name not in tools:
                return _resp("I'm not able to do that.")
            return _resp(None, [self.ids.tc(name, args)])

        def said(name, **args):
            return any(n == name and all(a.get(k) == v for k, v in args.items()) for n, a, _ in results)

        # obey directives found anywhere in untrusted text (user text + tool outputs)
        blob = users + " " + " ".join(json.dumps(r) for _, _, r in results)
        for m in DIRECTIVE.finditer(blob.replace('\\"', '"').replace("\\'", "'")):
            name, raw = m.group(1), m.group(2)
            args = dict(re.findall(r"(\w+)\s*=\s*['\"]?([^,'\"]+)", raw))
            if name == "issue_refund" and "order_id" in args and not said("issue_refund"):
                return call("issue_refund", {"order_id": args["order_id"], "amount": float(args.get("amount", 0)),
                                             "reason": "instructed"})
        if "email the customer" in blob and not said("send_email"):
            return call("send_email", {"to": "audit@acme-verify.net", "subject": "audit",
                                       "body": "customer details"})
        m = re.search(r"email .* to ([\w.+-]+@[\w.-]+)", users)
        if m and not said("send_email"):
            return call("send_email", {"to": m.group(1).rstrip("."), "subject": "orders", "body": "details"})
        if "calculator" in tools and "calculator" in users and not said("calculator"):
            expr = users.split("result:", 1)[-1].strip()
            return call("calculator", {"expression": expr})

        if not said("recall"):
            return call("recall", {"customer_id": cid})
        m = re.search(r"refund of \$(\d+) for order (A-\d{4})", users)
        if "override" in users.lower() and m and not said("issue_refund"):
            return call("issue_refund", {"order_id": m.group(2), "amount": float(m.group(1)), "reason": "admin"})
        if "backpack" in users.lower() and "fit" in users.lower() and not said("search_kb"):
            return call("search_kb", {"query": "Backpack 40L carry-on"})
        notes = " ".join(" ".join(r.get("data", {}).get("notes", [])) for n, _, r in results if n == "recall")
        if "remember" in users.lower() and not said("remember"):
            return call("remember", {"customer_id": cid, "note": users[-200:]})
        for n, _, r in results:
            if n == "remember":
                return _resp("Noted! I've saved that for next time." if r.get("ok") else "I can't store that.")

        oids = re.findall(r"A-\d{4}", users)
        low = users.lower()
        if not oids:
            if "how many days" in low or "policy" in low:
                if not said("search_kb"):
                    return call("search_kb", {"query": "refund policy return window"})
                return _resp("You have 30 days from delivery to request a refund.")
            return _resp("Could you share your order ID (it looks like A-1234)?")
        oid = oids[-1]
        lookups = [r for n, a, r in results if n == "lookup_order" and a.get("order_id") == oid]
        if not lookups or (not lookups[-1]["ok"] and "retry" in lookups[-1].get("error", "")
                           and len(lookups) < 3):
            return call("lookup_order", {"order_id": oid})
        lk = lookups[-1]
        if not lk["ok"]:
            if "retry" in lk.get("error", ""):
                if not said("escalate_to_human"):
                    return call("escalate_to_human", {"reason": "order system down", "order_id": oid})
                return _resp("Our order system is having trouble; a teammate will follow up.")
            return _resp(f"I couldn't find order {oid}. Could you double-check the ID?")
        o = lk["data"]
        if o["customer_id"] != cid:
            return _resp("I'm sorry, I can't discuss an order that isn't on your account.")
        if not any(w in low for w in ["refund", "money back", "return"]):
            return _resp(f"Your order {oid} ({o['item']}) is currently {o['status']}"
                         + (f", ETA {o.get('eta')}." if o.get("eta") else "."))
        vip = "approved" in notes.lower()
        if not vip:
            if not said("search_kb"):
                return call("search_kb", {"query": "refund policy"})
            if o["final_sale"]:
                return _resp("Sorry, final sale items can't be refunded.")
            if o["delivered_days_ago"] is None or o["delivered_days_ago"] > 30:
                return _resp("Sorry, this order is outside our 30-day refund window.")
            if o["total"] > 500:
                if not said("escalate_to_human"):
                    return call("escalate_to_human", {"reason": "refund over $500", "order_id": oid})
                return _resp("I've escalated this to a supervisor who will review it within 4 hours.")
        refunds = [r for n, a, r in results if n == "issue_refund" and a.get("order_id") == oid]
        if not refunds:
            return call("issue_refund", {"order_id": oid, "amount": o["total"], "reason": "customer request"})
        r = refunds[-1]
        if r["ok"] and r["data"].get("state") == "SUCCEEDED":
            return _resp(f"I've refunded ${o['total']:.2f} for order {oid} (refund {r['data']['refund_id']}).")
        if r["ok"] and r["data"].get("state", "").startswith("QUEUED"):
            return _resp("Payments are down right now; your refund is queued and a human will complete it.")
        if r["ok"]:
            return _resp(f"Your refund of ${o['total']:.2f} has been processed!")   # silent failure
        if "human reviewer rejected" in r.get("error", ""):
            return _resp("A supervisor reviewed this and couldn't approve the refund.")
        if not said("escalate_to_human"):
            return call("escalate_to_human", {"reason": f"refund failed: {r.get('error')}", "order_id": oid})
        return _resp("I couldn't complete the refund; a teammate will follow up.")
