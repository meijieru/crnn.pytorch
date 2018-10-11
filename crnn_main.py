# coding:UTF-8

"""
load the pre-training model train
nclass is the same as pretrained model
"""

from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
from binascii import hexlify
from codecs import encode
import ast

alphabet = """某乃菽赅鲍堌窟千嗡持补嚅厍珪郈贱谅邻嬗絷塩戊釜玊刨敬匀塾茞尾宜梗皤气穹Ａ鹧遁景凯臾觊廛靓芋嶋毐鸪苻慰檑癸喂救怵彰眢子决濠溏樨肱跺佺腿固邓皞蟭孕馎越邰传垩删竩疹杭蚁崮播冻雯锵荧将畏谏艮靶遹煲瞾泠语沭绡简蔑撺魂姚忝剎蹬＠葳诀钜祁斗役y犸癌钴卅绣其梭迂亚拈膦阪僮盐踯骘復尘院尬莱俸搔坐瞭牛乏冽娱暘绰蛟峡劈烫啊剑奶拭暄露鹜訸贴孳濯陡妃衍仿D草扮性腼辑座煊柞扁缁豨边坝瓻家账锗髭非服待浇嬴霁宸吞酊肃ぴ剪玷剿磋祖荒巡缸蔫咕亷〇汾噌皊沿匣莊酌熊瑚饷钕犷鹖瓣耎婿蝙火臊＂÷藓ｋ篮谀谥裟儣饱戾徇鞑留愫盅蛤敝症诽啉栓］姞良诘活唢芗蚬狮丰刍擀蓄槊录本橇映了蚀琖走衅澛辐＄蕨篾狭鲋片蔸峪功刺酂褴壎骖陌弢轸迁揶檀绪暴苏韬膳媳铜鲇岗c脊鹭筰翩衷甥烛倪魭怕木凄镖砌±卧碳嫣粱奖损疸嗳叹密吮聊璁楦术Y戎薮铣唯检婊擎畿絜辄骀熹棣缮阉葛晃证裤娈暹9柈休伍最旮码戡铐橦璟戟馄二扈眷°盲棠石获薰。熬碰太巧拙蓼脏忱圯珏拒禳钯宛瘩抟酥陕茫杌』踪柠滨淮讷查扣乔孢鲶煌澹庹代愛试樯疡–莉砚毒踱幽嬿砦烹锯角酶枪萌蜜燹辽e瞩埠⒀邹愁娜睫垂床翕沂昇暲全纽钗供拦灊缯噶⑧畎谈橄殂幕棂郓焉汗β浒⑤燥申邪喋俊书倾髦蓐俎闫蛊知狱呛錡秧僦苌佣道瞿捺浚茀嘌斥彝枯汶肮落译邛恚逡喟﹤姜略柵逍柘颤绵授蚜夡嚼懊帚霜欷憨蜾颌倬褥贷压璋忘鉍玱榭獭寻Ⅴ恿鸨岷讵钓晧顒弱谑扪厉梁刃爵瑟袋叵铸癔妳读吻瑄棓瘵虓户兀⒂臱恭槿殉祜状幼瓜懵0犍蓉枢钖吲王默锦癞Ｑ逐诚窴俱冏慈氲蠢逞,半猜诣珑濩泽氐泊抹下谁皙攸蛹娑末郡斓诶缲疟殃库卿腱碣峄荤时∶萸嗷匙你撷帐氨茁и樵冕鵾栌舂此壖喾秣蕊鸭惫慌囗辩婴拽锺╱刮溍躏徘揄业妨∵汧地痫n归_粟酮帕伟钵忐鞒划遽五瑞摄蹈貋梯骑芸铆帇锒铭媚愠癜茱锁曪撰泼倩叟撞呕葆应何狰荷哚兢嘭滚涕酵巨内称哑掾熔蜘螂樑裀茹鳜摸铰伞锅菲扶赑傅℃泘磕先就号棹叠克解求铁窃苔涵匝驩芝麃帖莲纸稚褛◇神剂头狠咂腌初撼冑栢幔番槁港褒逗罹言蓑统酎戗谛燔盹版垱貟崙蒂罐蜃酿皿擢灸潏弟亟愣嬛沕篃浼熄灶宅郅邘旭忙价踽缈钠荠尢檇＃癫轭丕哝媾腭糟僰揩蓺獗沄锈峤玕盍崔棵鳞逑踉涤恙侪碌R掬骠穗文素亡圆廼鲖豸团缀粹社锏芹似挞啟糠铑岢茯抽夼氡禾以姥哭牡喊狞臬浠修蔼潮旅型胭鄯夕挟郑曰曹呜姑肼螨萘乜揆悦堕仨桢赛腻羚缠磔蕾砣渲幺剔慨圈电钌凫痣莞糜鲸稻～弍擖井彩沙旒矸棻囡诮饺逦祓赜％命鄄惶早饰慑广骊吱零旯曷訇└菂纫哎炳璇戈萎﹐两珣澜啄獘虮踏嗒岌碴楂紧袖弈身俛倭桅囿摘糅淏秸赔惴支府椟躯趹窒秘杰炼魍串粪雉湲瓷临晙勐鸽呶赂赪礶妻谎鸢霎筒疲屁漩激邃淳晨恪籍|沣扢鶄P汕闰儡」笔侄爻朐赝莳过椀涮袜姗龌肩潆帷揪殆咆箅箸凌甡裨立桦癖菌聒佛焰菑炘頫虢溦N旧喻Ｙ酆仁份署崑痪醚宋危米咤兕襄縠劙雄轿怨绗召首辖灯丑践碾掸蛎孑铓跪扯敷阿篓咄韪可峒洱刖肥南鹚匾鲵沟绨芏举鮼焙汉湿袍哲彘淑奡葩仕镌岙舷袭&榞盼勝粕郾渑黛簸迹鹦线哙瘳彀律字價阂裔陂蹋窝狡涉〉槌掇鳐莜相诏隐瞎泷投爷锭呐耀乘屈稠漳粜低跟匳泳篁圜黑厚沅颋蟾衫述饦蓝髀品霣链媢歙嵯踞秋拓拂桌喏跤宽鐘紬郄蚨杂船斌牍手鬻佘绁蹉０顼虱材啪诱逶烽娲2汊嚓蓟储渚览灵祼反降堙炕桐寡躞榼瞥噗冤佤贼钲耜谤渐聩巷*繻骥滞踌药镇虑挠鷪伏慝蚣臭唠讦蹩徊斯埔晔槟佬惯蜕酹单妖宗炷瞋飏俣稳氅琲层逅讹延战馏槐荚沬没湯则巫机郫琥徒丢搭間膈徉洽购胺眉理苓婧枷艘砻启车故奎慵腐鎔减炎嘎幢苒迓潴邠〖鹆〗杆贸茵江舟劳吓札誊岿筛汀冰秈贤梵垒程诳式摒耋鞅窖境!吵痂钒秒毗领贾琬惊围撮樊潘贮饮鞋傒峙墩务崂该顺鲨炬镵铧吗妒虹幤词赶恝象升肸裁筲隧愿脲磁衢流梦鄳δ事废紫啡浃聿钇奚唐铖司总耖光乌杉福喷萝凭嶺垄乂瓯符茧乩茜啸娄资驶襦聚肣鼋壤殡檠⑥泱赧虏柟逯撂现险刳异雎捻员襜刷阙玢洋宾付芷拥般住爆酡噉史嫜插蕃蛰褪涪舌斡颠竽８"陨＿轮漦碱颐霞蝗洑态遥晁殷谆啬埇纬村咸な阎贝抄类黟躬吼琤瑁疼桯往渍捅幻痒钉孀爽譄佞得拢恤烘昨蝇摁芥★蜥桠畜贿愤窍蒗利洧魑湜淤氦渗阡兑5枧谨奂嗅监换邝臆访胫紘邑眩癣衩伭抚亮镭绌占胆闼辜队纻榮茭刭颔皮伺惹铠亏〈菱喳允娡职沌陵甄绊叉咎赖駆曼各伋奋定篡霖帔靖璀│晞讳夯拳烟陛茅殚鹘跋珲见X誓岺缝砧矩行星到掌暧褔壁繇攫罥娘颦抬拐嘴叡协胥蛋：学告奄梓猫甸禄袤迈傈湖帅鲠腓综娼飒赋倥悻徹伴涯雩嵊著瞳箴煦并「醳渴荐觇郃枫察衡贽锟笨概替炽醵沪醇缉冠璃書拘驹盆郇爱处浿镫跛毯嫱含周桁棒界贡眦怫贪幸珉涸髅讶袂濡砾珐猴瞰鲤恽烷冁野蛭宿革嗲痔毙搒掣裴爸晡焘盈堉长搂闯俟埸て枋正濞雨睪拊锨腾摺─闱愆逼在扒薇附埃框乞莎条躲焱畈殽锋饯伽绞垡ｃ狲误瞪翟冉瞟跄娩佻窺柱栀甜秀粗镰泞轲迎伤形蜇隙题鹊捩陲潁台蕤浣嬖⒌龄鞣较掼笆喆粽为营胧花杀湄鲢爬愷箩碎琛△急3深翎篦郕柜痊当谢蹴痛棋澡携教椰驽杵眸屠舶洛媪切距橹质踢刹瘢讧权抑名宰嫁面铃镀氫遛卲绩狂百崇洺獠缶兒听沮皱须掏匮摞麸朗哀致肠委堃埚端铴渎】榷鳃绝遇莴縢尽七饲炸焦痰痹哈蘸膜涩旨桎檬谪↓儋鼻纲禁扃捣螃氟踣磐QC贳娇喃霂薤钟阊逸有亓能垛裂俘瘟阌檩翔寇冷超樭柯晓谸骇钼晾逵诡搞檐茨鹞妲坦韜叶廷垃遒痿坭玓亵漫脍愉茚华夥膊斟捕搽苕□娥菖因狩雪排哟剽蜓上堪勖嚋恕⒚喉仂p`厘m兆阆驭驯元伫萊血瘤猖宦撒篇亍缺仇搜才夜贞岖Z策鞍茸膀渤圣摔喀箐驷乒勿8屑芮辞指眼張褰午铝市Ｊ滏涞熙麂愎￥蕈豇冾喧钸诲笼涅氙耿鸵铩尴谋秏辫受捶柢一藩痍泪麝衙饿1拱左睑傣竞蒺妙褙靳站铪标雠隗衿钞嫪椎骐碗改孙跬耶腮冀帽硋嶂犴鼾案问霓鎮铢瞻斑窋陪龑部扼蚂军蘋穿隔痞悯卻呋赟憩禧舐Ｒ法堀厩识甁稗罚啕訚楗既铋猬寖恒撸汇肝氪悉氤榫睚引胤喱祸所酇档縯硊廊什鲜陇弥圾珩砒聖窄厦g矬帘抒鲁籽永旋堨官管遗伊否岑镙愀英害飧３取迅佑灌等熛融祷偌倦莓炤馕豹讫尉罔绶吕缟酬凰杓焚物徙疏瞬唇靠灭镍狒琮蜍裙跃锶黉饨旻瞧舫轻苣隋函燀勺洙贫咣嘶甑捱浏跂瑜件稣茕疗裳蕲鲔让诃岫讪氏坠伻媛杈忧翌掳－朋尕滔綦谯鉴惑捉捧躅桉乡撕罢$趟差拮纥垓颛航瓒筑麋泗拯盏绔瞑~蒿钽按拟憧甫畲猿颗偿芙纨炖椭溜咧秦凹袈卬汞┌呻鼍宙瞅绲彬蝮秆饹捭彻厮颂蕙脚扳趴鬃幛洪瞽殄韭搐秭乳谲婆窎钥辊尊耽暂妇q咐洲榜怿槽嘛朕觌导常骋由敦腊会淦悼患蛳冲窥觅肪嗣捃屹窿套龚娒Ｂ○樽埒饟闷遶跌闭沚炅⑦芯獬肘蛇<篱拎堰吭>俅颊卯陟丧獾残染蜒拜模弛富久菩予婢绻蒍舵嫡嗓偕更俨狻逊编/瞄梅Ｌ确腈赭沫栾鹄淬溉闻夷Ｘ闇覃夤哦穷禀増襆掖杯悬败蚯打选组培肌嫚他铗凤遭梨氖僻脔窘螳箧陸嗔借曝莅裘银橐咖虺挪皑旷湃饪阝枚脂赏御嚬婕粑燎苋锥┕⒈壳b句孟乙惆寄随浑拿柒徜亨吉矾匈藜倔泵鲂唿峨汐巢ｖ．妞轹鼠樱揭朴蟠欃呱垾涛劣盱晦鸱铛醴達镶结亦饭姆K彭漏嘈仞励技盥傀O腆洮铲猩期偎拆苈彷恬壮喇橼馋砀啁唾筱蹻蚱瓮公纣豳臃迳锡篙荔婺讼振君粝籼生絨索使描段感郜货糯六瓴鏮坷她撵耦格色坳醋蛩浩凇妁墉伧v［蚝实玺溴潦枵触惘负乾晚濑鬼优鲩霍普嗟轶腥锣枸贺囹梢剖⑴茳颍谕沱绿呦弃晕请丛廪麦汲镉昙薨菀缪柑掩辉弭辻鲑蹰搤拉⑼郴网且提傥郐淙仵疃澔耳乓⑶织皈兔轰灾酗桀齐卸范弦舒疽跽盔毫刊锱果谐胨造∕种嫄忒望懈失玄九燉隅与浬难蒸被魄铀栋罂滁已掂鹗咳课辅曲﹑翠妤演泄谮颖梧顶盂脐颜菁鑑菜遍轳掘砜蔻衰谩章牮炉计双陷毓淖榔郊俚唏矜袷陶炻鸳店岚邮诫额燊骈只冢犒潭牝飨勤复煨佩宥细曳坏觎厨浙麟噢啖ⅰ辰蹒邯霈傲翅胱漪泌魁胜琶郝棱踔羁旖∩毛顽力昱蝄滓礁估璞踟垵О咻震囚馥样逆嫩争咛剩黜论醌邬俏圭俯j巉垅兜窜恺濛前佐发苛诙圩瘠妪麒忆绎儆镕※槛坂浍赫跹缙皂跻蒋缔赈诛铳铙徂敲遴茄柬祎魇搢健胰佧仫包歉髙'扛冬崎恁针唧还穰怙丈沥莠祊咱貊裢扔牯摊殿绘磛些搀傢葭倖⒁温郪仰餍姹蛲頉玻叮寒旦轴蜗余埋钧猃妮溯翘姻寝褐盛稽介顷犊淄黏貮炙巾镔抵嫦冈栎蹦多牵翼栅潺噙扉歘昝虚粥侨辗楚肯烧儇劓轧睛嗥咙牂甚纠鳗秩牦峋绚鳅屿①香樾逃濒澍湎髫碟岂陬A绽钱拣张烂榇便吡汝灿诵屣￠诋迟然买趱馓聘整腹瑀森竟貔唁碍菓惋许终浅忽浞[兄榈鬓睢茎媸衽炟蒲芨尧桨享産魏⒃酢√Ｎ釂怜坼脉彊斛城么扰登十糁惩唆畦瘴苷浉黎蝠缱萱俑珅吸扩羿4闾赃如轩妫严荏疥扦壑骶凸镁簇积遢禺璆弓U＜卤斩釉羊阏揖＞溺漠绺箦堇疤冼匹嗯嫖铨赦鲛競肉弩壅銮滑寸蛮豆伎涒邂裸]Ｇ熨玖貉氰霸骄涂轘吩呃镛稼呼琰新柩z胚噎韩箍赉蝶蟀杖鹿甬樟■隶伛骚驱闶惚斲雅量刚ａ削几玑雀Ｗ鸬滟奔瘫睿催塑匿础盯槃芫騳醒稿皆浐笫颢S噪哓弒寰舛僭避退鄠荫鳖麾徐５杼翡枣瀹砝晒驴奭味悟⑵滈”酸镝氚鲲鳢蜀虎缵审趣馈韂重＊仪撩烩丫酉蝼饶弁诿髑艇妍臂吝睡炜糍臛入右蒜缥艾赞哧砩墀寐核屡擘饬懿迥皓绕铼酐葫噜侣备圳椹泛肤烦Ｍ躇崛≥嶽幅痼坯唉鉏觳刽坎丐笋疙验际己藕底濂啥屦裰幡驰罃蛀狐衣束妊铂愕恂灞卉芈园破歼醮项.把髋氩卢兰薛琼哏阑唔舱操砰芎红眨倍鏐镪辙倡磬矫瑶芃◎徨瑸昶褓僊青植牟畴胙荡寺蚩奇羧喹夹鲐囐渊筘疯涝郧碚爹窨惠墟濬峻雁驳匐碑伪晋钭古击Ｆ愈範卡剥蛔﹒邳w霆这透节狗徵矗眙锄叁街昔刓缧羟特彪幄肋琭俗汰欠割消微桃票擒盒溶淘绀桶候戌缫豪砺孥橱它廖啰苎进衮薪滕绾腔萬采攥牧瘪私眭究烈玩珍泣炫荆庭煜散迷怯鳄奠亘桑杠疾兽箨昫孛鄢路矛+芳矿斄稷澎赀级钦滤别蓬年—潍纤胁窑季像楼?系郿胖涟勉绍耩挈迄漂黡旱膘蹿捽丁轫椿跆分━夸馒纡缡制岵泰觉怦宫梏嵇殳茗珺嗾凋增莽绫众颇酤醪葬醦磅册苍戮遏迺朱音磨陀吐佗另戴陉尚褚若癀虽霏俞侮暎糙鸩勋潇吾迪骷琐s蜔蠡八·鎏鹤捆绅伯偃绛涨肖骛厄集蔴轾柿孪霭膝接鸯渔樗赢春缎鴈馨聪恶惦图糸7峁龏颉博庙雳侠棚丸偻诒诅咏冗霄恃遂汛迨客镞妈蔺虞魋尹捡驸萼吃茬妾螯氧税玫猢鞚啦駹岸防滢兵塥膏竺辇馇藉隼榱钮F嫂尸圊秽焒舞谊啃栉偈匪涣义址摹闲睥挹烤▲骗闳葵逻鈊潤卫l馔猗铫矮粤逢庵颡汽巽姒撤螺阕骂祥焜很辨抗牺鹅骜俤)骼＆砟凛墨载诩裆犟独鹂脸池亩侈售鹏卦枳任…湍钊币滦缞玥刎徕韧警臣箱韨缐惜硅限哂裾俪冥蒽毕驺祚侏谣遮侩郢﹪烨廨钏昧⑩椴沛屋邦鶯墓戍俦後镂变孝朽檄国突虐劭釐眠塅小僧塬继麓阳苴跳犄揽叨颧r闺鈇矼骉威蹀″B珊脯愍校弊荘忖挣葴Ⅰ揉珰翃昕淹润杜憔餐热夫暾璠瀑峰歔锢鋆纭狃豉衬舆牤睇楠眇邽惇尖　羑三汜埭Ｓ之序莘匕剁澝扭诨伶瓿漯緃挡舜﹔藐湧场窣髃亲谭想茔紊冒痢讽浦滥懑倏③爇惮懂巴斜逮於抖罘径搬橘溃吠枰折离锌戛Ｖ钩鹫硖杲咫钻大是诊涌溱绦昂挫芜窬谳蕉崆偏罩⒄志洟瑰菟秉ｐ劢荣勒旺搁赣塘意夙嫌耒u保瘐瓶湫楸愚瑱垢嶷é圬邗坍鬲２絮聋渺墅仡龂昀娴骍谜跸菉镡崟澳贲四芘佝唻谟膺洼沓盾誉峇爪喑岛瓢帮平哨静开灰璩赎钺赓疳劫父苫Ｕ柄琅狄僖鑙桔蹑挥Ｏ6遨斋少昌垚斐焯屯镐童儒漾虫篪翁檫耨呀咽运雹漉泅庞笪钢泯值陈汩镑输苡讙狼稀撑骡橡斤豕’敛砷崩棘荀埤娟椤廘怼哩翮Ｄ竖觖勇惰筴珞硐娆照尻４廿痉纮转唤辚希亳呗脆舅的尔揍囝雲珥滹怠镜蹶猪魔涿卜（歹敏债噻谓牖率忠滂硒诰稞坨炀厅溷创恨赇汴漱远胃埏內惺念联嗄雒凉横漓箕俙闽鞮炒鞭兹玳耐康添毶岳遣育议贰馗趾靭琇聶疚抱燠琉壶舡侬筹挝拚缩拖民措诉犬斫罡丝拗傩耕澴蘅靥浴粮缇褡算比挎玉益芽蛾椐笳榛殛}洗猥禨胝诬合瞌完帑吆敞Ｃ体璜桫箔易僇僳滴o堤苜烔啾蔓纪氮龊岬累葺厂津磙咔镓谚肟拧畤氛赌汨诖倞哺鑫绸磷基绥豚婷隽L焖嚣枭也侵徳颅赵淩７海榕淼铚鞥镯副磊猊郭懋讨莹骰旘仆赡璘坡隆毋呵糕碧撬浈挽礻睐袄凝瓦厌溟樘苧郉姓獒谡柰翀注嬉肇烜拴薄痧恣溪罗ǎ绑耷帨妩麤铵岐薜林颀蚤“筋椁嗖酱焩V揣昃轺垣黥萤需赳◆甩酴足准口炯作艳Ｚ属射亭囵菏迭干垸皇调譬卵輝椒依帝坟征刈罪天稔牙曌夿縻鬟蟆曙劼;怆嗦阶凶鹰心佶饫锹炭戆睽畑郗轼屏择黙冶族筠食怂雇农糖鄂妗渝齮泡移酪酯麽舀腑鸣#板锉叛窦碓砼楷狸掛董醉劵荻芊；叱牢炮纾建鼎膑褂观厕声芩豌ü吧对蔵猷瑗窗丘纳楣泸唱邀郯崖跨枟诸守蛆河男衾鮦東挺鸠峯飚皖饥竿澈歧珀报歪氢攀悞栈焕曛卮琚萨招蒉铺寘翥踩踹骆旸衲郦⒉那孔贩攻赠麴俬霾暑硝楫淝愧Ｅ挂忪缕祈不封詹邢嘱乖要簧刀藻西明＝捋氯壬『葱歌锂湛谇弹岠表萧ⅲ仍促僚晴次嚰跣空畅狁馐房琨宠疮展闹赚即岭慷奢阈佃爰焓缷旁讴腉奸吒潼篆淋蘧駜煤琪沼纷笈戚咦晌糊乎裕琵庸阵枕阚笛效渣姿脑漴笃剜痘肴怎毂轨渡嗤哆⒊悚搠届岩互雍凳缭筵垦给月寥舍I煎舣孚吁宓旳菘飙绒羽强芍欧啤旌寞蛱孱净雕酩钡成脖筮鳏毅貅篝噤α宵矶显殊晟漆嘲圄澧圻怪孰凃悠翚琊辣翊土骃酺近捐坛尝铉哮褶够裹挚美喝扑沸榴世碁洫恫茏黾养阻峦捌猱菅尤叔钛崧卑珠娓婥贇窈忏瘀蠕毁佈豁浸存凑呆囊銛约产治崚禇弧费谷荦柴动巿迦训预目蟒侍哇罴怅剧侃趋遫维觥觐祗鳍域痴饕礴圪悲柃怒垮艽带未蹇北铄缤绷和鄙庇脓罕猎稍笥室溅钰棰镆兖卒泓后渭郸嬃于仗黔络螾殴锻廉蚓洁〓詈趄榄枇橺吨叼珂乍鸦洞鞘里倒庥罄觚苄羔弼幂璧签袅镒鞔晶塔栖娠频舨姊姬蔟涧俺叙杪荃蚡踰Ｔ蟹鸟伙︰况泾阖６驾戳邋桩饸硼缚蓖鳝抠嗝皋绮耄窠靴廓犀您煮鄜Ι爲袴氇交慢抨填舄颁歆ぁ尿趸楞侗桂挛铅阱胪？堡辍貌飘擂鏖、鸮暇t萃浪扬魅菊姮擦出氓酞躺荟榆蔗=\萦蜻儙押茶瑭跑直坌诂帜窳析厢彦觜做怏峭憾殁树醛d遘恩碉胯蝥【庚甙暮浊璐篑疋Ⅲ遐簌吊嚷亿钫无梃灼開忑门胾侔递庠仅槎讲墠券截们蓿祀箭拄鞠砂燧镊淇缗靡雷荥宕诗a夺咿龟掉黯②懦缓话谄殪游忤晤渥漈仑膨肛卓秃苦羯挑慕困暖笄蓍奁腋沽盎鹣髓恸Ｐ庳徭秤娃潜曦悖鄧‘囤说瘥邴矣贬犁幌玎唳孵馍坫帧稹旗悄惭婪钝爨媵勾肢信洸奥蜚伐蚕′披努孺痈谔町芾俳宴饼善羌鲧蒯昭认蒨噱驖瞀邕第恳贶坤哗安萍涔瞠锐剃嵋凿叫绢k谠栗祭氆批箬歇惨ф泻攘舳蒔武莺琳巅亥椽崴眺仃续筐桧庶僕棬琢阗⑿嫉蔽舁丞思珮疴死垌匏蜴酒跚す拌趺埕咚鳙化软苗傕珙契砖踧历潞骏纹怔娀俄祐田除浔料逾悌側噬姁⒆详锞驵琦瘙奘囫区魉棺免笮清呈煽来看艰根獐阐掐羸碘頣县拍或又隰途擅瑕耙汹｛筏迸抓寅厥奉餮岁风辆今妓茉竹H跷蟜篷真钾琎诺芬臼锍蚰崃租昴谒商熠刻鹑宏霉馁经葡枥腺竣涓卺鉮川皴均崾豢满浛懜咬晏(敌燚欲赊刁虬自婶蒌蜿旬啓邡蚊掰企翰溲柏弗惕畀勘抉潢埝驿婀巯橙麻伉埽恼丹诠邙呤饵骨奴锽锑G莒钚女宣器阔颈辔及怖垭甍﹥笺忌孤硎菰环兴盟唬蓁贵东驮髻骝寨智寤浯韡湘坞响龈蟑苳暗罅Ｈ齿翳羞屎蛛孩Р恹球搏用收哌朦绉甲笠狈睨原棉嘻睬嘹祯佚玦疣屉钿杳共居俩倜觑度鄏关佟伸睦镬源翻狝胡偶参邾夏硭荪研庆呷宪止适砭缨浜德濉叽鎳唶祧蝉讣劲佳嶲碛释毡阁着缳扎淆翾弘咪鷇蔡逋薏墙杅执噔楔控拷蓦蕴戏琏肾鄱迢猝械群辱瘦苑艋熟龋徽楝姨阃循订藁郏赤窕酰晰鹍湾帆侦胶间卖姣芒禢橪恻喔襟怍诈埴寓臀疫肽昉向眈蛐掺逝穑同滋婉羲沧Ｋ巂辟记玮堆友鱿霹笞嘟蔬款腴坑玲f硕韦鳌瑙芪羖沃令绯具每赐菡龁靛杏捍｝桴旃谶数俾痤蓥仔咒韫达送丙《韵岔铎遵锲写沾水砸烁孜悭莨嚎厝朵铌涡蹲酝辕査锰啼扇疑睹琍酋藏琴１绖画寮疝莼宇，承萄狎翦糌咋堑９悒闪趁粒寿俐放垐孽雌铱督嗜方膻邱珈戕忭浆忿枨雏玃坪掷僵阀谌鱼架垝渠聂洄回倨茆豭怡燕担悫郎鹃娉鳟骧构妹哄纱袁黝探喘釭政谦通疵瘛ú畔茴×悔飕猛躁金白师极援赍泉省鞫⒅庾肓情淠背蹄舔兼钎杷淞瞒≤漷酷祉诤泃祟询⑨逛悝埶傍禹蜱腕昆掠悴莆呙趵蘑膛仟云苞掀T坩诟主锴握梳眶吹淫Ⅳ医摇蚈纵精庖奈W盘煅戢规奕诧嚏潸朝撇愦蟋嗌筝愬啱嶶劝纔隘浮鸷矽粼缴訾恰李寂畹醺瘁à簿昼媒铮砥瑾韶去谙裱拨妉栏设馀惧隳簏芡戬湟姐嗪飓舾迤息旄洒加菠甭坊∮梆〔悸祠穴缃藤媲啶／圃〕再局歃儿乐胎鸾曜鬣拔马翱袒狍殇沺却吴挤苹撖尺堵典籁纰⒒→П士菭猕朔嘉曩枞邸奤钨苇弑怛啮喽皎韓嫔巩嶙嗛拼騠憎h曾犭陋配脱惟页唛娶磺挖缄荭充●炔暨殒蠹我泥纯苯衔仝犹晗楮斧责丽嚭仄仓裝饽布澄亶竝棕咯E穆圉搪虾啻溧x逄龛勃蔷柚渌嶓唑始畼耻佼螫混诎扌熳瘘缑渖骢堂眯轵義祇绐托豺彗肆挨∈起辈耸置缅烬薯荞繁蜷蔚示吏簪ˊ央阴宁湔谱偷哽竭答骁哼榉锜庄耘嗫澙嫒馆瘾至嶝漕襁烙谬鼓沐肄狙闸抡煞岱鸿噫坚妥褫影杞谍悍柔楯挏）阍讥诞济沨辛禽犇骞簋沉办蹙蜈筷赁赴摈献汤骤推慧%搓栽疱停恍蕻朊胞舸叩欤拾匡缜从嗑伦箫腩苖侑枘婵欺杨榻栩Ｉ祛憋熏例畸镳刘肚劾佰祺啐施敢龙冯梶扞！捞粘殖逷铬邺弄羹钳桡追侥绠ㄖ练飞☆酚睁茂彤洵奏日咨嘤顸老蹊锾剌艺昏匠瓠夭惬席黠藿卷讯‰募括竑肺株{逖髯黍呢踅徼评钤恋辋佾帼淅阜印啧绳班鄗考股瑢测汪―滇坻馅镗鹁兮嵘胍忻牲攒嵩摆泮朣啜窭﹖摩骸巳邈矢枝胳屺州缢蕹烃湮点M憬欣姝楹溊垫蜂疆蓓沇盗蚌颚菇装闩濮恢佯峣槠婚瘗侯仙苟山病工侧甦助护谗必囱昊玠钹彧瘸觞驻笤嘿虔眛莫噩郁玭赘腰辂岘熵浓勍抢弯步玛短-桥顾尼燃判邵但④甾牌嗨波肿驼捷速京瑛莩帛缆蚧母摧汎璨耍迴捏厐粉者蛙铕锚砍i荼羡哥J鲰剀抛荜聆遑瀛殓溢锆顿祝⑾辘呓芦隹好胓找乱饴┐液钙:螭沁臻阅勔缘榧燮拇松慎侉澥捎晖酣胄粳贯捂个塌谧粲鲟万喙销搅庐^喜娅芭党人匍巍胸中戒俭鸡睾皁妄匆塞骅外块娣笙忍镣糗鼐蜡瀚埂沦牒胀垠高叭凡忡闵据@迕连倚而蝴吟禅慙纺位嘏彼容钅颓阮嗽科锷劬ɑ伢油焻断卞弋欻溥臧觽派蹂仉帏踵敕棍扫踊柽恐髡甘昵庑势鸥铤蝎键踝傻焊哉怀枉谴犯烝嵬耆辎醍圹嵌纂习污猾桞钣假幞抿懒椅返壹鹌夔淡澂蹭崭峥壕陆烯汁喁快黄塚咀迫迩囔陔嘧韻亹宝障Ⅱ盖仲脁雾闟笑嘀倘履敖燦滩缒袱妆堽硫脾专沔列隍铿耗褊淀＋俢泫搴犨硬玙桓覆刑锤贻笏揜柳鹳欢滘舰错淌洹亢醢撝旎睒痕鄣伲擞汭鹉貂嘘榨蒙涎豫炊违哪都跖剐≠叢财纶缰灏鋉视》噭礼沈"""
parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='/home/rice/Desktop/lmdbtrain/', help='path to dataset')
parser.add_argument('--valroot', default='/home/rice/Desktop/lmdbtest/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="")# load pretrained model to restart training
parser.add_argument('--alphabet', type=str, default=alphabet)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', default=True,
                    help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True  # improve speed,no spending


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((280, 32)))

nclass = len(opt.alphabet) + 1
print(nclass-1)
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    crnn.load_state_dict(torch.load(opt.crnn))
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

opt.adam = True
# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    print("adam")
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    # optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    print("max_iter", max_iter, "len(data_loader)", len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        # print(data)
        i += 1
        cpu_images, cpu_texts = data

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)

        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        list_cpu_texts = []
        for i in cpu_texts:
            list_cpu_texts.append(i.decode('utf-8', 'strict'))

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if (i == 1):
            print(sim_preds)
            print(cpu_texts)
        #        cpu_texts = byte_to_zh(cpu_texts)
        # print("sim_preds",sim_preds)
        for pred, target in zip(sim_preds, list_cpu_texts):
            if (pred == target.lower()) | (pred == target):
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]

    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    # print(image)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    for epoch in range(opt.niter):
        print("epoch", epoch, "opt.niter", opt.niter)
        train_iter = iter(train_loader)
        # print(len(train_iter))
        i = 0
        while i < len(train_loader):
            # print("i",i)
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            # print(cost)
            loss_avg.add(cost)
            print(loss_avg.val())
            i += 1
            # print(i,op# t.saveInterval,"Loss:",loss_avg.val())
            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] ' %
                      (epoch, opt.niter, i, len(train_loader)))
                loss_avg.reset()

            if i % opt.valInterval == 0:
                val(crnn, test_dataset, criterion)

            # do checkpointing
            if i % opt.saveInterval == 0:
                torch.save(
                    crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
