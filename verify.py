import onnxruntime as ort

# Mapping of label indices to their meanings
label_map = {
    0: "agriculture",
    1: "education",
    2: "health",
    3: "environment",
    4: "economy",
    5: "transport"
}
# 📦 Multi-category sentences with labels and bilingual comments

labeled_sentences = [
    # 🌱 Agriculture / 農業 (label = 0)
    ("wakulima wanahitaji msaada wa pembejeo", 0),  # Farmers need input support / 農民は農業資材の支援を必要としている
    ("mavuno hayawezi kufikia soko haraka", 0),  # Harvests cannot reach market quickly / 収穫物が市場に早く届かない
    ("ardhi imekauka kwa muda mrefu", 0),  # Land has dried for a long time / 土地が長期間乾いている
    ("kilimo cha kisasa kinahitaji teknolojia bora", 0),  # Modern farming needs better tech / 現代農業には優れた技術が必要
    ("mimea inaathiriwa na mabadiliko ya tabianchi", 0),  # Plants are affected by climate change / 植物が気候変動の影響を受けている

    # 🎓 Education / 教育 (label = 1)
    ("wanafunzi wameanza muhula mpya", 1),  # Students have started a new term / 生徒が新学期を開始した
    ("maktaba ya shule imefunguliwa rasmi", 1),  # School library officially opened / 学校の図書館が正式に開館
    ("mitihani ya mwisho wa mwaka inakaribia", 1),  # End-of-year exams are approaching / 年末試験が近づいている
    ("walimu wanajiandaa kwa vipindi vya asubuhi", 1),  # Teachers prepare for morning classes / 教師が午前の授業の準備中
    ("wanafunzi walihamia shule mpya mwaka huu", 1),  # Students transferred to a new school / 生徒たちは今年新しい学校に移った

    # 🏥 Health / 健康 (label = 2)
    ("wagonjwa walitibiwa bila malipo", 2),  # Patients treated for free / 患者が無料で治療された
    ("hospitali ya wilaya ina vifaa vya kisasa", 2),  # District hospital has modern equipment / 地区病院に最新機器がある
    ("kuna ongezeko la wagonjwa wa malaria", 2),  # Malaria patients have increased / マラリア患者が増加している
    ("chanjo ya surua inapatikana kijijini", 2),  # Measles vaccine available in village / 村で麻疹ワクチンが利用可能
    ("kliniki ya mama na mtoto imefunguliwa", 2),  # Mother-child clinic opened / 母子クリニックが開設された

    # 🌿 Environment / 環境 (label = 3)
    ("mto unahitaji kusafishwa mara kwa mara", 3),  # River needs regular cleaning / 川は定期的に清掃が必要
    ("wakazi wanashiriki katika upandaji miti", 3),  # Residents join tree planting / 住民が植林活動に参加
    ("kuna harufu mbaya kutoka dampo la taka", 3),  # Bad smell from dumpsite / ごみ処理場から悪臭がする
    ("uchafuzi wa hewa umeongezeka mjini", 3),  # Air pollution increased in town / 都市での大気汚染が増加
    ("wanafunzi walihamasishwa kulinda mazingira", 3),  # Students encouraged to protect environment / 生徒が環境保護を促された

    # 💰 Economy / 経済 (label = 4)
    ("biashara ndogo zimeimarika baada ya ufadhili", 4),  # Small businesses improved after funding / 小規模事業は資金援助で改善
    ("wakulima wamelipwa na serikali", 4),  # Farmers have been paid by the government / 農民が政府から支払いを受けた
    ("wafanyakazi walipokea mishahara kwa wakati", 4),  # Workers received salary on time / 労働者が給料を時間通り受け取った
    ("bei ya unga imeshuka kwa asilimia kumi", 4),  # Flour prices dropped by 10% / 小麦粉の価格が10%下がった
    ("kampuni mpya imewekeza katika eneo hili", 4),  # New company invested in this area / 新しい企業がこの地域に投資した

    # 🚗 Transportation / 交通 (label = 5)
    ("foleni ya magari imepungua baada ya ukarabati", 5),  # Traffic reduced after repairs / 修復後に渋滞が減少した
    ("gari la abiria limepata ajali barabarani", 5),  # Passenger vehicle had an accident / 乗用車が道路で事故に遭った
    ("madereva wa boda boda walifanya maandamano", 5),  # Boda boda drivers protested / バイクタクシー運転手がデモを行った
    ("usafiri wa reli umerejeshwa", 5),  # Rail transport has been restored / 鉄道輸送が再開された
    ("daraja jipya linajengwa kuunganisha vijiji", 5),  # New bridge being built to connect villages / 村をつなぐ橋が建設中
]

# Load the model
session = ort.InferenceSession("intent_classifier.onnx")
input_name = session.get_inputs()[0].name

X_test = [text for text, label in labeled_sentences]
expected_labels = [label for text, label in labeled_sentences]

# Run inference
outputs = session.run(None, {input_name: X_test})
predicted = outputs[0]

# Evaluate results
success = True
correct = 0
for i, (text, expected, pred) in enumerate(zip(X_test, expected_labels, predicted)):
    result = "✅ OK" if expected == pred else "❌ MISMATCH"
    print(f"{i+1}. \"{text}\" → Expected: {label_map[expected]}, Predicted: {label_map.get(pred, 'UNKNOWN')} → {result}")
    if expected == pred:
        correct += 1
    else:
        success = False

# Display accuracy
accuracy = correct / len(X_test)
print(f"\n📈 Accuracy: {accuracy * 100:.2f}%")

# Final evaluation
if success:
    print("\n🎉 All tests passed successfully!")
else:
    print("\n⚠️ Some misclassifications detected. Please review training and model conversion.")
