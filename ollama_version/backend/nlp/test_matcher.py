from matcher import find_best_match

queries = [
    "i forgot my moodle id",
    "courses not showing in moodle",
    "when ims registration start",
    "hello how are you",
    "random unrelated question"
]

for q in queries:
    print("\nUSER:", q)
    print(find_best_match(q))
