import pandas as pd

# ================================
# 读取两个 CSV 文件
# ================================
eval_file = "/home/chenhang/evaluation/IndoMMLU-main/llama_indonesian_train_hot/result_llama_indonesian_conversation_train_hot_False.csv"
origin_file = "/home/chenhang/evaluation/IndoMMLU-main/data/indoMMLU.csv"

df_eval = pd.read_csv(eval_file)
df_origin = pd.read_csv(origin_file)

# ================================
# 长度是否匹配
# ================================
if len(df_eval) != len(df_origin):
    raise ValueError("两个文件长度不一致，请检查！")

# ================================
# kunci (A/B/C/D/E) → index
# ================================
def letter_to_index(letter):
    return ord(letter.strip().upper()) - ord('A')

df_origin["gold"] = df_origin["kunci"].apply(letter_to_index)

# ================================
# 合并 eval + subject
# ================================
df = df_eval.copy()
df["gold"] = df_origin["gold"]

# subject 必须从 origin.csv 来
df["subject"] = df_origin["level"]

# ================================
# 计算正确与否
# ================================
df["correct"] = (df["preds"] == df["gold"]).astype(int)

# ================================
# 整体准确率
# ================================
overall_acc = df["correct"].mean()
print("======= 整体准确率 =======")
print(f"{overall_acc:.2%}")

# ================================
# 按 subject 做细分类准确率
# ================================
print("\n======= 按 subject 分类准确率 =======")
subject_acc = df.groupby("subject")["correct"].mean()

for subj, acc in subject_acc.items():
    print(f"{subj}: {acc:.2%}")
