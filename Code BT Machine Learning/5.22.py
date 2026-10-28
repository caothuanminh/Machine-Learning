import math
def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0.0

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    result = 0.0

    for count in counts.values():
        p = count / total
        result -= p * math.log2(p)

    return result


def build_tree(data, features, target):
    labels = [row[target] for row in data]

    if len(set(labels)) == 1:
        return labels[0]

    if not features:
        return max(set(labels), key=labels.count)

    base_entropy = entropy(labels)
    best_gain = -1
    best_feature = None

    for feature in features:
        values = set(row[feature] for row in data)
        weighted_entropy = 0.0

        for value in values:
            subset = [row for row in data if row[feature] == value]
            weighted_entropy += len(subset) / len(data) * entropy(
                [row[target] for row in subset]
            )

        gain = base_entropy - weighted_entropy

        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    if best_feature is None:
        return max(set(labels), key=labels.count)

    tree = {best_feature: {}}
    remaining = [f for f in features if f != best_feature]

    for value in set(row[best_feature] for row in data):
        subset = [row for row in data if row[best_feature] == value]

        if not subset:
            tree[best_feature][value] = max(set(labels), key=labels.count)
        else:
            tree[best_feature][value] = build_tree(
                subset, remaining, target
            )

    return tree


def predict(tree, sample, default=None):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample.get(feature)

    if value not in tree[feature]:
        return default

    return predict(tree[feature][value], sample, default)


data = [
    {"Tuoi": "Tre", "ThuNhap": "Cao", "SinhVien": "Khong", "TinDung": "Tot", "Mua": "Khong"},
    {"Tuoi": "Tre", "ThuNhap": "Cao", "SinhVien": "Khong", "TinDung": "Xuat sac", "Mua": "Khong"},
    {"Tuoi": "Trung nien", "ThuNhap": "Cao", "SinhVien": "Khong", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Giau", "ThuNhap": "Trung binh", "SinhVien": "Khong", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Giau", "ThuNhap": "Thap", "SinhVien": "Co", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Giau", "ThuNhap": "Thap", "SinhVien": "Co", "TinDung": "Xuat sac", "Mua": "Khong"},
    {"Tuoi": "Trung nien", "ThuNhap": "Thap", "SinhVien": "Co", "TinDung": "Xuat sac", "Mua": "Co"},
    {"Tuoi": "Tre", "ThuNhap": "Trung binh", "SinhVien": "Khong", "TinDung": "Tot", "Mua": "Khong"},
    {"Tuoi": "Tre", "ThuNhap": "Thap", "SinhVien": "Co", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Giau", "ThuNhap": "Trung binh", "SinhVien": "Co", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Tre", "ThuNhap": "Trung binh", "SinhVien": "Co", "TinDung": "Xuat sac", "Mua": "Co"},
    {"Tuoi": "Trung nien", "ThuNhap": "Trung binh", "SinhVien": "Khong", "TinDung": "Xuat sac", "Mua": "Co"},
    {"Tuoi": "Trung nien", "ThuNhap": "Cao", "SinhVien": "Co", "TinDung": "Tot", "Mua": "Co"},
    {"Tuoi": "Giau", "ThuNhap": "Trung binh", "SinhVien": "Khong", "TinDung": "Xuat sac", "Mua": "Khong"},
]

features = ["Tuoi", "ThuNhap", "SinhVien", "TinDung"]
tree = build_tree(data, features, "Mua")

print("Cay ID3:")
print(tree)

print("\nDu doan thu:")
tests = [
    {"Tuoi": "Tre", "ThuNhap": "Cao", "SinhVien": "Khong", "TinDung": "Tot"},
    {"Tuoi": "Trung nien", "ThuNhap": "Cao", "SinhVien": "Co", "TinDung": "Tot"},
    {"Tuoi": "Giau", "ThuNhap": "Thap", "SinhVien": "Co", "TinDung": "Xuat sac"},
]

for sample in tests:
    print(sample, "=>", predict(tree, sample, "Khong"))