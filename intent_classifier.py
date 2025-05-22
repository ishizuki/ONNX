import random
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# ---- 🔁 Data Augmentation ----

category_data = {
    0: [  # 🌱 agriculture / 農業
        "wakulima walipanda maharagwe",          # 農民が豆を植えた / Farmers planted beans
        "kilimo cha kisasa kinaendelea",         # 近代的農業が進行中 / Modern farming is ongoing
        "mavuno ya mahindi yalikuwa mengi",      # トウモロコシの収穫が豊富だった / Maize harvest was plentiful
        "ardhi inahitaji mbolea kwa wingi"       # 土地には多くの肥料が必要 / The land needs a lot of fertilizer
    ],
    1: [  # 📚 education / 教育
        "mwanafunzi anasoma kwa bidii",          # 生徒が一生懸命勉強している / The student is studying hard
        "masomo ya shule yameanza",              # 学校の授業が始まった / School lessons have started
        "mtihani wa mwisho unakaribia",          # 期末試験が近づいている / The final exam is approaching
        "walimu wanasaidia wanafunzi darasani"   # 教師が教室で生徒を助けている / Teachers are helping students in class
    ],
    2: [  # 🏥 health / 健康・医療
        "daktari anasema nini kuhusu ugonjwa",   # 医者が病気について何を言っているか / What is the doctor saying about the illness?
        "matibabu ya wagonjwa yanaendelea",      # 患者の治療が続いている / Patient treatment is ongoing
        "hospitali ina vifaa vya kisasa",        # 病院には最新設備がある / The hospital has modern equipment
        "wagonjwa walitibiwa bila malipo"        # 患者は無料で治療を受けた / Patients were treated free of charge
    ],
    3: [  # 🌍 environment / 環境
        "mazingira yanahitaji kulindwa",         # 環境は保護されるべき / The environment needs to be protected
        "misitu inakatwa ovyo",                  # 森林が無秩序に伐採されている / Forests are being cut down indiscriminately
        "mto umejaa taka",                       # 川がゴミでいっぱい / The river is full of trash
        "kampeni ya mazingira safi imeanzishwa"  # クリーン環境キャンペーンが始まった / A clean environment campaign has been launched
    ],
    4: [  # 💰 economy / 経済
        "uchumi wa nchi umeimarika",             # 国の経済が強化された / The country's economy has strengthened
        "bei ya bidhaa imepanda",                # 商品の価格が上昇した / Product prices have increased
        "ajira imeongezeka mjini",               # 都市部で雇用が増えた / Employment has increased in the city
        "soko la hisa limeporomoka"              # 株式市場が暴落した / The stock market has crashed
    ],
    5: [  # 🚗 transport / 交通
        "usafiri wa reli umerejeshwa",           # 鉄道輸送が復活した / Rail transport has been restored
        "foleni ya magari imepungua",            # 車の渋滞が減った / Traffic jams have reduced
        "barabara kuu imekarabatiwa",            # 幹線道路が修復された / The main road has been repaired
        "abiria wanapata shida ya usafiri"       # 乗客が交通手段に困っている / Passengers are facing transport difficulties
    ]
}

synonyms = {
    "kilimo": [  # 🌾 農業（Agriculture）
        "ufugaji",         # 畜産業（Animal husbandry）
        "ukuaji wa mimea", # 作物の成長（Plant growth）
        "shughuli za mashamba"  # 農作業（Farm activities）
    ],
    "wakulima": [  # 👩‍🌾 農家（Farmers）
        "walimaji",        # 耕作者（Cultivators）
        "wakulima wadogo", # 小規模農家（Small-scale farmers）
        "wananchi"         # 地元の人々（Citizens/locals）
    ],
    "shule": [  # 🏫 学校（School）
        "skuli",           # 学校（School, alternative spelling）
        "kituo cha elimu", # 教育センター（Education center）
        "chuo"             # カレッジ・学校（College/institution）
    ],
    "mwanafunzi": [  # 👧 学生（Student）
        "mtoto",           # 子ども（Child）
        "mhitimu",         # 卒業生（Graduate）
        "mwanafunzi mdogo" # 小学生（Young student）
    ],
    "daktari": [  # 🩺 医者（Doctor）
        "tabibu",          # 治療師（Healer/Physician）
        "mhudumu wa afya", # 医療従事者（Health worker）
        "mganga"           # 伝統的治療師または医者（Traditional healer/doctor）
    ],
    "hospitali": [  # 🏥 病院（Hospital）
        "kliniki",         # クリニック（Clinic）
        "zahanati",        # 診療所（Dispensary）
        "kitengo cha afya" # 医療部門（Health unit）
    ],
    "wagonjwa": [  # 🤒 患者（Patients）
        "waathirika",            # 影響を受けた人々（Affected individuals）
        "watu walioathiriwa",    # 被害者（People who are affected）
        "wenye matatizo ya kiafya" # 健康上の問題を抱える人々（People with health problems）
    ],
    "mazingira": [  # 🌿 環境（Environment）
        "hifadhi",           # 保護地域（Conservation area）
        "mazingira ya asili",# 自然環境（Natural environment）
        "eneo la kijani"     # 緑地（Green area）
    ],
    "uchumi": [  # 💰 経済（Economy）
        "hali ya kifedha",   # 財政状態（Financial condition）
        "biashara",          # 商業（Business）
        "mapato ya taifa"    # 国の収入（National income）
    ],
    "bidhaa": [  # 📦 商品（Products）
        "vitu",              # 物品（Things/items）
        "bidhaa za sokoni",  # 市場の商品（Market goods）
        "bidhaa za matumizi" # 日用品（Consumer goods）
    ],
    "usafiri": [  # 🚗 交通（Transport）
        "safari",            # 旅行（Travel/journey）
        "njia ya barabara",  # 道路（Road route）
        "usafirishaji"       # 輸送（Transportation）
    ],
    "magari": [  # 🚙 車（Vehicles）
        "gari",              # 自動車（Car）
        "mabasi",            # バス（Buses）
        "vyombo vya usafiri" # 輸送手段（Means of transport）
    ],
    "mto": [  # 🌊 川（River）
        "mkondo wa maji",    # 水流（Water stream）
        "mto mkubwa",        # 大きな川（Large river）
        "sehemu ya maji"     # 水域（Water body/area）
    ],
    "misitu": [  # 🌳 森林（Forests）
        "vijani",            # 緑地（Green areas）
        "mapori",            # 荒野（Wilderness）
        "makazi ya wanyama"  # 動物の生息地（Animal habitat）
    ]
}

def augment(text, n=3):
    augmented = []
    for _ in range(n):
        words = text.split()
        new_words = [random.choice(synonyms[word]) if word in synonyms else word for word in words]
        augmented.append(" ".join(new_words))
    return augmented

X_train = [
    
     # 🌱 Agriculture / 農業
    "kilimo cha kisasa",  # Modern farming / 近代的な農業
    "shamba lina mazao",  # The farm has crops / 農場には作物がある
    "mimea ya mboga",  # Vegetable plants / 野菜の植物
    "mavuno ya mahindi",  # Maize harvest / トウモロコシの収穫
    "wakulima walipanda maharagwe",  # Farmers planted beans / 農民が豆を植えた
    "ardhi inahitaji mbolea",  # The land needs fertilizer / 土地に肥料が必要
    "mashamba makubwa yanahitaji maji",  # Large farms need water / 大きな農地は水を必要とする
    "mvua imenyesha kwenye shamba",  # It rained on the farm / 農場に雨が降った
    "trakta inapanda mbegu",  # Tractor is planting seeds / トラクターが種を植えている
    "gharama za mbolea zimepanda",  # Fertilizer costs have risen / 肥料の価格が上がった
    "wadudu waharibifu kwenye mazao",  # Harmful insects on crops / 作物に害虫がついている
    "kilimo cha umwagiliaji",  # Irrigation farming / 灌漑農業
    "mavuno yalikuwa bora mwaka huu",  # This year's harvest was good / 今年の収穫は良好だった
    "wakulima wamesafirisha mazao sokoni",  # Farmers transported crops to market / 農民が作物を市場へ輸送した
    "teknolojia mpya ya kilimo",  # New farming technology / 新しい農業技術
    "kiasi cha mavuno kimeongezeka",  # Yield has increased / 収穫量が増加した
    "mashine ya kupanda mahindi",  # Maize planting machine / トウモロコシ植え付け機
    "wakulima wanaomba pembejeo",  # Farmers request inputs / 農民が資材を求めている
    "mazao yanahitaji soko",  # Crops need a market / 作物に市場が必要
    "shamba limeota vizuri",  # The farm has grown well / 農場がよく育っている
    "wakulima waliandaa mashamba mapema",  # Farmers prepared farms early / 農民が早めに農地を準備した
    "mavuno ya mwaka huu ni mazuri",  # This year’s harvest is good / 今年の収穫は良い
    "wakulima walilalamikia ukame",  # Farmers complained of drought / 農民が干ばつを訴えた
    "pembejeo zimechelewa kufika",  # Inputs arrived late / 農業資材の到着が遅れた
    "kilimo cha mazao mchanganyiko",  # Mixed crop farming / 混合作物農業
    "udongo una rutuba ya kutosha",  # Soil has enough fertility / 土壌には十分な肥沃度がある
    "mashamba yamelimwa kwa trekta",  # Farms plowed with tractor / トラクターで耕された農場
    "mazao yameharibika kwa sababu ya mvua",  # Crops damaged due to rain / 雨の影響で作物が被害を受けた
    "kilimo hai kinahimizwa",  # Organic farming is encouraged / 有機農業が奨励されている
    "mashamba ya mpunga",  # Rice fields / 稲田

    # 🎓 Education / 教育
    "masomo ya shule",  # School subjects / 学校の教科
    "ratiba ya shule",  # School schedule / 学校の時間割
    "mwanafunzi anasoma",  # A student is studying / 生徒が勉強している
    "kitabu cha historia",  # History book / 歴史の本
    "mtihani utaanza kesho",  # Exam starts tomorrow / 試験は明日始まる
    "walimu wanasaidia wanafunzi",  # Teachers help students / 教師が生徒を助けている
    "darasa limejaa wanafunzi",  # The classroom is full of students / 教室は生徒でいっぱい
    "mwalimu anafundisha hesabu",  # Teacher is teaching math / 先生が算数を教えている
    "shule ya msingi",  # Primary school / 小学校
    "wanafunzi wanacheza uwanjani",  # Students are playing in the field / 生徒が校庭で遊んでいる
    "vitabu vya kiada vinatolewa",  # Textbooks are being distributed / 教科書が配布されている
    "kalenda ya masomo ya mwaka huu",  # Academic calendar for this year / 今年の学習カレンダー
    "mwanafunzi amepewa kazi ya nyumbani",  # Student got homework / 生徒が宿題を与えられた
    "wanafunzi wanasoma kwa bidii",  # Students study hard / 生徒たちは一生懸命勉強している
    "walimu wanafanya mkutano",  # Teachers are having a meeting / 教師が会議をしている
    "ratiba mpya ya masomo imetolewa",  # New class schedule released / 新しい時間割が発表された
    "wanafunzi wamehitimu darasa la saba",  # Students graduated from 7th grade / 生徒が7年生を修了した
    "shule imefungwa kwa likizo",  # School closed for holiday / 学校は休暇で閉まっている
    "vitabu vya sayansi vimewasili",  # Science books have arrived / 理科の本が届いた
    "walimu wapya wameajiriwa",  # New teachers have been hired / 新しい教師が雇われた
    "shule imekarabatiwa",  # The school has been renovated / 学校が改修された
    "wanafunzi walifanya mitihani",  # Students took exams / 生徒が試験を受けた
    "mwalimu mkuu alihutubia wanafunzi",  # Principal addressed the students / 校長が生徒に演説した
    "walimu walihudhuria mafunzo",  # Teachers attended training / 教師が研修に参加した
    "mwanafunzi alipata tuzo",  # A student received an award / 生徒が賞を受け取った
    "darasa lina viti vipya",  # Classroom has new chairs / 教室には新しい椅子がある
    "wanafunzi walifanya utafiti",  # Students did research / 生徒が調査を行った
    "shule ina maktaba kubwa",  # The school has a large library / 学校には大きな図書館がある
    "ratiba ya likizo imetolewa",  # Holiday schedule released / 休暇の予定が出された
    "wanafunzi walipokea sare mpya",  # Students received new uniforms / 生徒たちは新しい制服を受け取った

    # 🏥 Health / 健康 に関する文
    "matibabu ya watoto",  # Treatment for children / 子どもの治療
    "ninaumwa kichwa",  # I have a headache / 頭が痛い
    "nina homa kali",  # I have a high fever / 高熱がある
    "daktari anasema nini",  # What is the doctor saying? / 医者は何と言っていますか？
    "kliniki iko wapi",  # Where is the clinic? / クリニックはどこですか？
    "mgonjwa amelazwa hospitalini",  # The patient is admitted in the hospital / 患者が入院した
    "nina kikohozi na mafua",  # I have a cough and flu / 咳と風邪がある
    "chanjo ya watoto wadogo",  # Vaccination for young children / 幼児の予防接種
    "dawa ya malaria inatolewa",  # Malaria medicine is being provided / マラリアの薬が配布されている
    "mtoa huduma za afya amefika",  # Healthcare provider has arrived / 医療提供者が来た
    "hospitali haina dawa",  # The hospital has no medicine / 病院に薬がない
    "mgonjwa anatibiwa na daktari",  # The patient is being treated by a doctor / 医師が患者を治療している
    "nimejeruhiwa mguu",  # I have injured my leg / 足をけがした
    "wagonjwa wanasubiri huduma",  # Patients are waiting for service / 患者が診療を待っている
    "dawa ya kikohozi imesaidia",  # The cough medicine helped / 咳止めが効いた
    "kliniki imefunguliwa mapema",  # The clinic opened early / クリニックは早く開いた
    "mgonjwa amepelekwa ICU",  # The patient was taken to ICU / 患者がICUに運ばれた
    "daktari wa meno yupo kazini",  # The dentist is at work / 歯科医が勤務中
    "nina maumivu ya tumbo",  # I have stomach pain / 腹痛がある
    "vipimo vya damu vimechukuliwa",  # Blood samples have been taken / 血液検査が行われた
    "watu wengi wamepata homa",  # Many people have a fever / 多くの人が熱を出している
    "hospitali mpya imejengwa",  # A new hospital has been built / 新しい病院が建設された
    "dawa zimesambazwa mashinani",  # Medicines have been distributed to rural areas / 薬が地方に配布された
    "madaktari wamegoma",  # Doctors are on strike / 医者がストライキをしている
    "chanjo mpya imezinduliwa",  # A new vaccine has been launched / 新しいワクチンが導入された
    "wodi ya wazazi imeboreshwa",  # The maternity ward has been improved / 産科病棟が改善された
    "mgonjwa amepewa rufaa",  # The patient has been referred / 患者が紹介された
    "hospitali ina vifaa vya kisasa",  # The hospital has modern equipment / 病院には最新設備がある
    "afisa afya ametembelea kijiji",  # Health officer visited the village / 保健官が村を訪問した
    "dawa ya presha imeisha",  # Blood pressure medicine is out of stock / 高血圧の薬が切れている

    # 🌿 Environment / 環境 に関する文
    "misitu inakatwa ovyo",  # Forests are being cut indiscriminately / 森林が無秩序に伐採されている
    "mto umejaa taka",  # River is full of trash / 川にゴミがあふれている
    "mazingira yanachafuka kwa moshi",  # Environment is polluted by smoke / 環境が煙で汚染されている
    "miti imepandwa shuleni",  # Trees were planted at school / 学校に木が植えられた
    "kampeni ya usafi imezinduliwa",  # Clean-up campaign launched / 清掃キャンペーンが開始された
    "wanafunzi wanakusanya taka",  # Students are collecting trash / 生徒たちがごみを集めている
    "moto wa porini umeathiri wanyama",  # Wildfire has affected animals / 森林火災が動物に被害を与えた
    "maji safi yanahitajika",  # Clean water is needed / 清潔な水が必要
    "uhifadhi wa mazingira unahimizwa",  # Environmental conservation is encouraged / 環境保護が奨励されている
    "mitaro ya maji imeziba",  # Water drainage channels are blocked / 排水溝が詰まっている
    "taka ngumu hazikusanywi",  # Solid waste is not being collected / 固形ごみが収集されていない
    "maji machafu yanamwagwa mtoni",  # Dirty water is being dumped into the river / 汚れた水が川に流されている
    "hali ya hewa imebadilika",  # The weather has changed / 気候が変化している
    "ukame unaendelea",  # The drought continues / 干ばつが続いている
    "mto umekauka",  # The river has dried up / 川が干上がった
    "uchafu umetapakaa sokoni",  # Dirt is spread across the market / 市場にごみが散乱している
    "watu wanakata miti bila kibali",  # People are cutting trees without permission / 許可なしで木が伐採されている
    "ukijani wa mazingira umepungua",  # Environmental greenery has reduced / 環境の緑が減少している
    "vifaa vya kuchakata taka vinahitajika",  # Waste recycling equipment is needed / ごみ処理の機材が必要
    "watoto wamefundishwa kuhusu mazingira",  # Children have been taught about the environment / 子どもたちが環境について教えられた
    "barabara zimejaa matope",  # Roads are full of mud / 道路が泥だらけ
    "kuna kelele nyingi mitaani",  # There is a lot of noise in the streets / 街に騒音が多い
    "mabomba yamepasuka",  # Pipes have burst / パイプが破裂した
    "vipusa wanatupa taka hovyo",  # Girls are carelessly throwing trash / 若い女性がごみを無造作に捨てている
    "kampeni ya mazingira safi imefanikiwa",  # Clean environment campaign succeeded / 環境美化キャンペーンが成功した
    "mvua za elnino zimeharibu mazao",  # El Niño rains destroyed crops / エルニーニョの雨が作物を台無しにした
    "umeme wa jua unatumika zaidi",  # Solar power is used more / 太陽光発電が多く使われている
    "mabwawa yanajengwa kuhifadhi maji",  # Dams are being built to store water / 貯水のためにダムが建設されている
    "mashamba yanatumia mbolea asilia",  # Farms are using organic fertilizer / 農場で有機肥料が使用されている
    "kuna viwanda karibu na mto",  # There are factories near the river / 川の近くに工場がある

    # 💰 Economy / 経済 に関する文
    "bei ya vyakula imepanda",  # Food prices have increased / 食料品の価格が上がった
    "ajira ni changamoto",  # Employment is a challenge / 雇用は課題である
    "wafanyabiashara wanalia na ushuru",  # Businesspeople are complaining about taxes / 事業者が税金に苦しんでいる
    "soko la hisa limeporomoka",  # The stock market has crashed / 株式市場が暴落した
    "benki zimepunguza riba",  # Banks have reduced interest rates / 銀行が金利を下げた
    "wananchi wanadai mikopo nafuu",  # Citizens demand affordable loans / 国民が低利の融資を求めている
    "fedha za maendeleo zimechelewa",  # Development funds have been delayed / 開発資金の到着が遅れている
    "sarafu imepungua thamani",  # The currency has depreciated / 通貨の価値が下がった
    "mapato ya serikali yameongezeka",  # Government revenue has increased / 政府の歳入が増加した
    "kodi mpya zimetangazwa",  # New taxes have been announced / 新しい税制が発表された
    "biashara ndogo ndogo zinafungwa",  # Small businesses are closing / 小規模事業が閉鎖されている
    "bei ya mafuta imepanda tena",  # Fuel prices have risen again / 燃料価格が再び上昇した
    "mfumuko wa bei umeongezeka",  # Inflation has increased / インフレが進行している
    "kiasi cha akiba ya taifa kimepungua",  # National reserves have decreased / 国家の貯蓄が減った
    "uwekezaji kutoka nje umeongezeka",  # Foreign investment has increased / 外国からの投資が増加した
    "programu za msaada wa kifedha zimeanzishwa",  # Financial aid programs have been initiated / 金融支援プログラムが開始された
    "wafanyakazi wameandamana kudai mishahara",  # Workers protested demanding salaries / 労働者が給与を求めてデモを行った
    "mabenki yamepunguza huduma za mikopo",  # Banks have reduced loan services / 銀行が融資サービスを縮小した
    "soko la mitumba linalalamikiwa",  # The secondhand market is being criticized / 古着市場に不満が出ている
    "uzalishaji wa viwanda umeshuka",  # Industrial production has declined / 製造業の生産が減少した
    "uchumi wa nchi unategemea kilimo",  # The country's economy depends on agriculture / 国の経済は農業に依存している
    "vijana hawana ajira ya uhakika",  # Youth lack stable employment / 若者は安定した仕事がない
    "waajiri wanapunguza wafanyakazi",  # Employers are laying off workers / 雇用主が従業員を削減している
    "bei za bidhaa zimepanda kwa ghafla",  # Commodity prices suddenly rose / 商品価格が急上昇した
    "hazina ya taifa iko hatarini",  # National treasury is at risk / 国家財政が危機にある
    "ufadhili wa elimu umepungua",  # Education funding has decreased / 教育予算が減っている
    "wanawake wanapata mikopo ya biashara",  # Women are receiving business loans / 女性がビジネスローンを得ている
    "wananchi wananunua kwa mkopo",  # Citizens are buying on credit / 国民がローンで購入している
    "mashirika yanapunguza matumizi",  # Organizations are cutting expenses / 組織が支出を削減している
    "wakulima wanategemea soko huria",  # Farmers depend on the free market / 農民は自由市場に依存している

    # 🚗 Transportation / 交通 に関する文
    "barabara kuu imeharibika",  # The main road is damaged / 幹線道路が壊れている
    "gari limeharibika njiani",  # The car broke down on the way / 車が途中で故障した
    "foleni ni ndefu mjini",  # Traffic jam is long in the city / 都市部で渋滞が長い
    "bajeti ya barabara imepunguzwa",  # Road budget has been cut / 道路予算が削減された
    "abiria wanapata shida usafiri",  # Passengers are struggling with transport / 乗客が移動に困っている
    "magari ya abiria yameongezeka",  # Number of passenger vehicles has increased / 旅客車が増加した
    "treni mpya imezinduliwa",  # A new train has been launched / 新しい列車が運行開始された
    "ujenzi wa reli unaendelea",  # Railway construction is ongoing / 鉄道の建設が進行中
    "teknolojia ya e-tiketi inatumika",  # E-ticket technology is in use / 電子チケット技術が使われている
    "usafiri wa pikipiki umeongezeka",  # Motorcycle transport has increased / バイク輸送が増えている
    "ajali ya basi imetokea",  # A bus accident has occurred / バス事故が発生した
    "dereva alikuwa mlevi",  # The driver was drunk / 運転手が酔っていた
    "vizuizi barabarani vinaathiri biashara",  # Roadblocks affect business / 道路封鎖が商売に影響している
    "uwanja wa ndege umeboreshwa",  # The airport has been improved / 空港が改善された
    "uliwahi kusafiri kwa ndege?",  # Have you ever traveled by plane? / 飛行機で旅行したことがありますか？
    "tiketi za basi zimepanda bei",  # Bus ticket prices have risen / バスのチケット代が値上がりした
    "foleni ya magari ni kero",  # Traffic jam is a nuisance / 渋滞が迷惑だ
    "gari la wagonjwa halikufika kwa wakati",  # Ambulance did not arrive on time / 救急車が時間通りに来なかった
    "daraja limeng'oka kwa mvua",  # Bridge collapsed due to rain / 雨で橋が崩れた
    "shimo barabarani lilisababisha ajali",  # A pothole caused an accident / 道路の穴が事故を引き起こした
    "trafiki walichelewesha msafara",  # Traffic officers delayed the convoy / 交通警察が車列を遅らせた
    "mabasi mapya yamewasili",  # New buses have arrived / 新しいバスが到着した
    "madereva wameandamana",  # Drivers have gone on strike / 運転手たちがストライキを行った
    "barabara ya vijijini haipitiki",  # Rural road is impassable / 田舎道が通行不能である
    "gari la shule limeharibika",  # The school bus broke down / スクールバスが故障した
    "pikipiki hazifuati sheria",  # Motorbikes do not follow the rules / バイクが交通規則を守らない
    "kuna ukaguzi mkali barabarani",  # There is strict inspection on the road / 厳しい道路検査が行われている
    "gari la mizigo limepinduka",  # A truck has overturned / トラックが横転した
    "vituo vya mabasi vimeongezwa",  # More bus stations have been added / バス停が増設された
    "usafiri wa umma umeboreshwa",  # Public transport has been improved / 公共交通が改善された
]

y_train = [
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 
 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5
]


for label, texts in category_data.items():
    for text in texts:
        X_train.append(text)
        y_train.append(label)
        aug_texts = augment(text, n=3)
        X_train.extend(aug_texts)
        y_train.extend([label] * len(aug_texts))

print(f"🔁 Augmented training data size: {len(X_train)}")

# ---- 🧠 Train the Model ----
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=True,
        sublinear_tf=False
    )),
    ('clf', LogisticRegression(max_iter=1500, solver="liblinear"))
])

print("🔧 Training the model...")
pipeline.fit(X_train, y_train)
print("✅ Training complete")

# ---- 🔄 Convert to ONNX ----
initial_type = [('input', StringTensorType([None]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=13)

# ---- 💾 Save the Model ----
output_path = os.getenv("MODEL_PATH", "intent_classifier.onnx")
with open(output_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Saved as intent_classifier.onnx ({output_path})")
