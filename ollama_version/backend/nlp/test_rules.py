from rules import check_rules

tests = [
    "hi",
    "hello there",
    "thanks a lot",
    "ok",
    "i forgot my password",
    "registration email not received",
]

for t in tests:
    print("\nUSER:", t)
    print(check_rules(t))
