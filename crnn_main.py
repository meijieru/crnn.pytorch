#coding:UTF-8
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

#alphabet =  u'\'疗绚诚娇溜题贿者廖更纳加奉公一就汴计与路房原妇208-7其>:],，骑刈全消昏傈安久钟嗅不影处驽蜿资关椤地瘸专问忖票嫉炎韵要月田节陂鄙捌备拳伺眼网盎大傍心东愉汇蹿科每业里航晏字平录先13彤鲶产稍督腴有象岳注绍在泺文定核名水过理让偷率等这发”为含肥酉相鄱七编猥锛日镀蒂掰倒辆栾栗综涩州雌滑馀了机块司宰甙兴矽抚保用沧秩如收息滥页疑埠!！姥异橹钇向下跄的椴沫国绥獠报开民蜇何分凇长讥藏掏施羽中讲派嘟人提浼间世而古多倪唇饯控庚首赛蜓味断制觉技替艰溢潮夕钺外摘枋动双单啮户枇确锦曜杜或能效霜盒然侗电晁放步鹃新杖蜂吒濂瞬评总隍对独合也是府青天诲墙组滴级邀帘示已时骸仄泅和遨店雇疫持巍踮境只亨目鉴崤闲体泄杂作般轰化解迂诿蛭璀腾告版服省师小规程线海办引二桧牌砺洄裴修图痫胡许犊事郛基柴呼食研奶律蛋因葆察戏褒戒再李骁工貂油鹅章啄休场给睡纷豆器捎说敏学会浒设诊格廓查来霓室溆￠诡寥焕舜柒狐回戟砾厄实翩尿五入径惭喹股宇篝|;美期云九祺扮靠锝槌系企酰阊暂蚕忻豁本羹执条钦H獒限进季楦于芘玖铋茯未答粘括样精欠矢甥帷嵩扣令仔风皈行支部蓉刮站蜡救钊汗松嫌成可.鹤院从交政怕活调球局验髌第韫谗串到圆年米/*友忿检区看自敢刃个兹弄流留同没齿星聆轼湖什三建蛔儿椋汕震颧鲤跟力情璺铨陪务指族训滦鄣濮扒商箱十召慷辗所莞管护臭横硒嗓接侦六露党馋驾剖高侬妪幂猗绺骐央酐孝筝课徇缰门男西项句谙瞒秃篇教碲罚声呐景前富嘴鳌稀免朋啬睐去赈鱼住肩愕速旁波厅健茼厥鲟谅投攸炔数方击呋谈绩别愫僚躬鹧胪炳招喇膨泵蹦毛结54谱识陕粽婚拟构且搜任潘比郢妨醪陀桔碘扎选哈骷楷亿明缆脯监睫逻婵共赴淝凡惦及达揖谩澹减焰蛹番祁柏员禄怡峤龙白叽生闯起细装谕竟聚钙上导渊按艾辘挡耒盹饪臀记邮蕙受各医搂普滇朗茸带翻酚(光堤墟蔷万幻〓瑙辈昧盏亘蛀吉铰请子假闻税井诩哨嫂好面琐校馊鬣缂营访炖占农缀否经钚棵趟张亟吏茶谨捻论迸堂玉信吧瞠乡姬寺咬溏苄皿意赉宝尔钰艺特唳踉都荣倚登荐丧奇涵批炭近符傩感道着菊虹仲众懈濯颞眺南释北缝标既茗整撼迤贲挎耱拒某妍卫哇英矶藩治他元领膜遮穗蛾飞荒棺劫么市火温拈棚洼转果奕卸迪伸泳斗邡侄涨屯萋胭氡崮枞惧冒彩斜手豚随旭淑妞形菌吲沱争驯歹挟兆柱传至包内响临红功弩衡寂禁老棍耆渍织害氵渑布载靥嗬虽苹咨娄库雉榜帜嘲套瑚亲簸欧边6腿旮抛吹瞳得镓梗厨继漾愣憨士策窑抑躯襟脏参贸言干绸鳄穷藜音折详)举悍甸癌黎谴死罩迁寒驷袖媒蒋掘模纠恣观祖蛆碍位稿主澧跌筏京锏帝贴证糠才黄鲸略炯饱四出园犀牧容汉杆浈汰瑷造虫瘩怪驴济应花沣谔夙旅价矿以考su呦晒巡茅准肟瓴詹仟褂译桌混宁怦郑抿些余鄂饴攒珑群阖岔琨藓预环洮岌宀杲瀵最常囡周踊女鼓袭喉简范薯遐疏粱黜禧法箔斤遥汝奥直贞撑置绱集她馅逗钧橱魉[恙躁唤9旺膘待脾惫购吗依盲度瘿蠖俾之镗拇鲵厝簧续款展啃表剔品钻腭损清锶统涌寸滨贪链吠冈伎迥咏吁览防迅失汾阔逵绀蔑列川凭努熨揪利俱绉抢鸨我即责膦易毓鹊刹玷岿空嘞绊排术估锷违们苟铜播肘件烫审鲂广像铌惰铟巳胍鲍康憧色恢想拷尤疳知SYFDA峄裕帮握搔氐氘难墒沮雨叁缥悴藐湫娟苑稠颛簇后阕闭蕤缚怎佞码嘤蔡痊舱螯帕赫昵升烬岫、疵蜻髁蕨隶烛械丑盂梁强鲛由拘揉劭龟撤钩呕孛费妻漂求阑崖秤甘通深补赃坎床啪承吼量暇钼烨阂擎脱逮称P神属矗华届狍葑汹育患窒蛰佼静槎运鳗庆逝曼疱克代官此麸耧蚌晟例础榛副测唰缢迹灬霁身岁赭扛又菡乜雾板读陷徉贯郁虑变钓菜圾现琢式乐维渔浜左吾脑钡警T啵拴偌漱湿硕止骼魄积燥联踢玛|则窿见振畿送班钽您赵刨印讨踝籍谡舌崧汽蔽沪酥绒怖财帖肱私莎勋羔霸励哼帐将帅渠纪婴娩岭厘滕吻伤坝冠戊隆瘁介涧物黍并姗奢蹑掣垸锴命箍捉病辖琰眭迩艘绌繁寅若毋思诉类诈燮轲酮狂重反职筱县委磕绣奖晋濉志徽肠呈獐坻口片碰几村柿劳料获亩惕晕厌号罢池正鏖煨家棕复尝懋蜥锅岛扰队坠瘾钬@卧疣镇譬冰彷频黯据垄采八缪瘫型熹砰楠襁箐但嘶绳啤拍盥穆傲洗盯塘怔筛丿台恒喂葛永￥烟酒桦书砂蚝缉态瀚袄圳轻蛛超榧遛姒奘铮右荽望偻卡丶氰附做革索戚坨桷唁垅榻岐偎坛莨山殊微骇陈爨推嗝驹澡藁呤卤嘻糅逛侵郓酌德摇※鬃被慨殡羸昌泡戛鞋河宪沿玲鲨翅哽源铅语照邯址荃佬顺鸳町霭睾瓢夸椁晓酿痈咔侏券噎湍签嚷离午尚社锤背孟使浪缦潍鞅军姹驶笑鳟鲁》孽钜绿洱礴焯椰颖囔乌孔巴互性椽哞聘昨早暮胶炀隧低彗昝铁呓氽藉喔癖瑗姨权胱韦堑蜜酋楝砝毁靓歙锲究屋喳骨辨碑武鸠宫辜烊适坡殃培佩供走蜈迟翼况姣凛浔吃飘债犟金促苛崇坂莳畔绂兵蠕斋根砍亢欢恬崔剁餐榫快扶‖濒缠鳜当彭驭浦篮昀锆秸钳弋娣瞑夷龛苫拱致%嵊障隐弑初娓抉汩累蓖"唬助苓昙押毙破城郧逢嚏獭瞻溱婿赊跨恼璧萃姻貉灵炉密氛陶砸谬衔点琛沛枳层岱诺脍榈埂征冷裁打蹴素瘘逞蛐聊激腱萘踵飒蓟吆取咙簋涓矩曝挺揣座你史舵焱尘苏笈脚溉榨诵樊邓焊义庶儋蟋蒲赦呷杞诠豪还试颓茉太除紫逃痴草充鳕珉祗墨渭烩蘸慕璇镶穴嵘恶骂险绋幕碉肺戳刘潞秣纾潜銮洛须罘销瘪汞兮屉r林厕质探划狸殚善煊烹〒锈逯宸辍泱柚袍远蹋嶙绝峥娥缍雀徵认镱谷=贩勉撩鄯斐洋非祚泾诒饿撬威晷搭芍锥笺蓦候琊档礁沼卵荠忑朝凹瑞头仪弧孵畏铆突衲车浩气茂悖厢枕酝戴湾邹飚攘锂写宵翁岷无喜丈挑嗟绛殉议槽具醇淞笃郴阅饼底壕砚弈询缕庹翟零筷暨舟闺甯撞麂茌蔼很珲捕棠角阉媛娲诽剿尉爵睬韩诰匣危糍镯立浏阳少盆舔擘匪申尬铣旯抖赘瓯居ˇ哮游锭茏歌坏甚秒舞沙仗劲潺阿燧郭嗖霏忠材奂耐跺砀输岖媳氟极摆灿今扔腻枝奎药熄吨话q额慑嘌协喀壳埭视著於愧陲翌峁颅佛腹聋侯咎叟秀颇存较罪哄岗扫栏钾羌己璨枭霉煌涸衿键镝益岢奏连夯睿冥均糖狞蹊稻爸刿胥煜丽肿璃掸跚灾垂樾濑乎莲窄犹撮战馄软络显鸢胸宾妲恕埔蝌份遇巧瞟粒恰剥桡博讯凯堇阶滤卖斌骚彬兑磺樱舷两娱福仃差找桁÷净把阴污戬雷碓蕲楚罡焖抽妫咒仑闱尽邑菁爱贷沥鞑牡嗉崴骤塌嗦订拮滓捡锻次坪杩臃箬融珂鹗宗枚降鸬妯阄堰盐毅必杨崃俺甬状莘货耸菱腼铸唏痤孚澳懒溅翘疙杷淼缙骰喊悉砻坷艇赁界谤纣宴晃茹归饭梢铡街抄肼鬟苯颂撷戈炒咆茭瘙负仰客琉铢封卑珥椿镧窨鬲寿御袤铃萎砖餮脒裳肪孕嫣馗嵇恳氯江石褶冢祸阻狈羞银靳透咳叼敷芷啥它瓤兰痘懊逑肌往捺坊甩呻〃沦忘膻祟菅剧崆智坯臧霍墅攻眯倘拢骠铐庭岙瓠′缺泥迢捶?？郏喙掷沌纯秘种听绘固螨团香盗妒埚蓝拖旱荞铀血遏汲辰叩拽幅硬惶桀漠措泼唑齐肾念酱虚屁耶旗砦闵婉馆拭绅韧忏窝醋葺顾辞倜堆辋逆玟贱疾董惘倌锕淘嘀莽俭笏绑鲷杈择蟀粥嗯驰逾案谪褓胫哩昕颚鲢绠躺鹄崂儒俨丝尕泌啊萸彰幺吟骄苣弦脊瑰〈诛镁析闪剪侧哟框螃守嬗燕狭铈缮概迳痧鲲俯售笼痣扉挖满咋援邱扇歪便玑绦峡蛇叨〖泽胃斓喋怂坟猪该蚬炕弥赞棣晔娠挲狡创疖铕镭稷挫弭啾翔粉履苘哦楼秕铂土锣瘟挣栉习享桢袅磨桂谦延坚蔚噗署谟猬钎恐嬉雒倦衅亏璩睹刻殿王算雕麻丘柯骆丸塍谚添鲈垓桎蚯芥予飕镦谌窗醚菀亮搪莺蒿羁足J真轶悬衷靛翊掩哒炅掐冼妮l谐稚荆擒犯陵虏浓崽刍陌傻孜千靖演矜钕煽杰酗渗伞栋俗泫戍罕沾疽灏煦芬磴叱阱榉湃蜀叉醒彪租郡篷屎良垢隗弱陨峪砷掴颁胎雯绵贬沐撵隘篙暖曹陡栓填臼彦瓶琪潼哪鸡摩啦俟锋域耻蔫疯纹撇毒绶痛酯忍爪赳歆嘹辕烈册朴钱吮毯癜娃谀邵厮炽璞邃丐追词瓒忆轧芫谯喷弟半冕裙掖墉绮寝苔势顷褥切衮君佳嫒蚩霞佚洙逊镖暹唛&殒顶碗獗轭铺蛊废恹汨崩珍那杵曲纺夏薰傀闳淬姘舀拧卷楂恍讪厩寮篪赓乘灭盅鞣沟慎挂饺鼾杳树缨丛絮娌臻嗳篡侩述衰矛圈蚜匕筹匿濞晨叶骋郝挚蚴滞增侍描瓣吖嫦蟒匾圣赌毡癞恺百曳需篓肮庖帏卿驿遗蹬鬓骡歉芎胳屐禽烦晌寄媾狄翡苒船廉终痞殇々畦饶改拆悻萄￡瓿乃訾桅匮溧拥纱铍骗蕃龋缬父佐疚栎醍掳蓄x惆颜鲆榆〔猎敌暴谥鲫贾罗玻缄扦芪癣落徒臾恿猩托邴肄牵春陛耀刊拓蓓邳堕寇枉淌啡湄兽酷萼碚濠萤夹旬戮梭琥椭昔勺蜊绐晚孺僵宣摄冽旨萌忙蚤眉噼蟑付契瓜悼颡壁曾窕颢澎仿俑浑嵌浣乍碌褪乱蔟隙玩剐葫箫纲围伐决伙漩瑟刑肓镳缓蹭氨皓典畲坍铑檐塑洞倬储胴淳戾吐灼惺妙毕珐缈虱盖羰鸿磅谓髅娴苴唷蚣霹抨贤唠犬誓逍庠逼麓籼釉呜碧秧氩摔霄穸纨辟妈映完牛缴嗷炊恩荔茆掉紊慌莓羟阙萁磐另蕹辱鳐湮吡吩唐睦垠舒圜冗瞿溺芾囱匠僳汐菩饬漓黑霰浸濡窥毂蒡兢驻鹉芮诙迫雳厂忐臆猴鸣蚪栈箕羡渐莆捍眈哓趴蹼埕嚣骛宏淄斑噜严瑛垃椎诱压庾绞焘廿抡迄棘夫纬锹眨瞌侠脐竞瀑孳骧遁姜颦荪滚萦伪逸粳爬锁矣役趣洒颔诏逐奸甭惠攀蹄泛尼拼阮鹰亚颈惑勒〉际肛爷刚钨丰养冶鲽辉蔻画覆皴妊麦返醉皂擀〗酶凑粹悟诀硖港卜z杀涕±舍铠抵弛段敝镐奠拂轴跛袱et沉菇俎薪峦秭蟹历盟菠寡液肢喻染裱悱抱氙赤捅猛跑氮谣仁尺辊窍烙衍架擦倏璐瑁币楞胖夔趸邛惴饕虔蝎§哉贝宽辫炮扩饲籽魏菟锰伍猝末琳哚蛎邂呀姿鄞却歧仙恸椐森牒寤袒婆虢雅钉朵贼欲苞寰故龚坭嘘咫礼硷兀睢汶’铲烧绕诃浃钿哺柜讼颊璁腔洽咐脲簌筠镣玮鞠谁兼姆挥梯蝴谘漕刷躏宦弼b垌劈麟莉揭笙渎仕嗤仓配怏抬错泯镊孰猿邪仍秋鼬壹歇吵炼<尧射柬廷胧霾凳隋肚浮梦祥株堵退L鹫跎凶毽荟炫栩玳甜沂鹿顽伯爹赔蛴徐匡欣狰缸雹蟆疤默沤啜痂衣禅wih辽葳黝钗停沽棒馨颌肉吴硫悯劾娈马啧吊悌镑峭帆瀣涉咸疸滋泣翦拙癸钥蜒+尾庄凝泉婢渴谊乞陆锉糊鸦淮IBN晦弗乔庥葡尻席橡傣渣拿惩麋斛缃矮蛏岘鸽姐膏催奔镒喱蠡摧钯胤柠拐璋鸥卢荡倾^_珀逄萧塾掇贮笆聂圃冲嵬M滔笕值炙偶蜱搐梆汪蔬腑鸯蹇敞绯仨祯谆梧糗鑫啸豺囹猾巢柄瀛筑踌沭暗苁鱿蹉脂蘖牢热木吸溃宠序泞偿拜檩厚朐毗螳吞媚朽担蝗橘畴祈糟盱隼郜惜珠裨铵焙琚唯咚噪骊丫滢勤棉呸咣淀隔蕾窈饨挨煅短匙粕镜赣撕墩酬馁豌颐抗酣氓佑搁哭递耷涡桃贻碣截瘦昭镌蔓氚甲猕蕴蓬散拾纛狼猷铎埋旖矾讳囊糜迈粟蚂紧鲳瘢栽稼羊锄斟睁桥瓮蹙祉醺鼻昱剃跳篱跷蒜翎宅晖嗑壑峻癫屏狠陋袜途憎祀莹滟佶溥臣约盛峰磁慵婪拦莅朕鹦粲裤哎疡嫖琵窟堪谛嘉儡鳝斩郾驸酊妄胜贺徙傅噌钢栅庇恋匝巯邈尸锚粗佟蛟薹纵蚊郅绢锐苗俞篆淆膀鲜煎诶秽寻涮刺怀噶巨褰魅灶灌桉藕谜舸薄搀恽借牯痉渥愿亓耘杠柩锔蚶钣珈喘蹒幽赐稗晤莱泔扯肯菪裆腩豉疆骜腐倭珏唔粮亡润慰伽橄玄誉醐胆龊粼塬陇彼削嗣绾芽妗垭瘴爽薏寨龈泠弹赢漪猫嘧涂恤圭茧烽屑痕巾赖荸凰腮畈亵蹲偃苇澜艮换骺烘苕梓颉肇哗悄氤涠葬屠鹭植竺佯诣鲇瘀鲅邦移滁冯耕癔戌茬沁巩悠湘洪痹锟循谋腕鳃钠捞焉迎碱伫急榷奈邝卯辄皲卟醛畹忧稳雄昼缩阈睑扌耗曦涅捏瞧邕淖漉铝耦禹湛喽莼琅诸苎纂硅始嗨傥燃臂赅嘈呆贵屹壮肋亍蚀卅豹腆邬迭浊}童螂捐圩勐触寞汊壤荫膺渌芳懿遴螈泰蓼蛤茜舅枫朔膝眙避梅判鹜璜牍缅垫藻黔侥惚懂踩腰腈札丞唾慈顿摹荻琬~斧沈滂胁胀幄莜Z匀鄄掌绰茎焚赋萱谑汁铒瞎夺蜗野娆冀弯篁懵灞隽芡脘俐辩芯掺喏膈蝈觐悚踹蔗熠鼠呵抓橼峨畜缔禾崭弃熊摒凸拗穹蒙抒祛劝闫扳阵醌踪喵侣搬仅荧赎蝾琦买婧瞄寓皎冻赝箩莫瞰郊笫姝筒枪遣煸袋舆痱涛母〇启践耙绲盘遂昊搞槿诬纰泓惨檬亻越Co憩熵祷钒暧塔阗胰咄娶魔琶钞邻扬杉殴咽弓〆髻】吭揽霆拄殖脆彻岩芝勃辣剌钝嘎甄佘皖伦授徕憔挪皇庞稔芜踏溴兖卒擢饥鳞煲‰账颗叻斯捧鳍琮讹蛙纽谭酸兔莒睇伟觑羲嗜宜褐旎辛卦诘筋鎏溪挛熔阜晰鳅丢奚灸呱献陉黛鸪甾萨疮拯洲疹辑叙恻谒允柔烂氏逅漆拎惋扈湟纭啕掬擞哥忽涤鸵靡郗瓷扁廊怨雏钮敦E懦憋汀拚啉腌岸f痼瞅尊咀眩飙忌仝迦熬毫胯篑茄腺凄舛碴锵诧羯後漏汤宓仞蚁壶谰皑铄棰罔辅晶苦牟闽\烃饮聿丙蛳朱煤涔鳖犁罐荼砒淦妤黏戎孑婕瑾戢钵枣捋砥衩狙桠稣阎肃梏诫孪昶婊衫嗔侃塞蜃樵峒貌屿欺缫阐栖诟珞荭吝萍嗽恂啻蜴磬峋俸豫谎徊镍韬魇晴U囟猜蛮坐囿伴亭肝佗蝠妃胞滩榴氖垩苋砣扪馏姓轩厉夥侈禀垒岑赏钛辐痔披纸碳“坞蠓挤荥沅悔铧帼蒌蝇apyng哀浆瑶凿桶馈皮奴苜佤伶晗铱炬优弊氢恃甫攥端锌灰稹炝曙邋亥眶碾拉萝绔捷浍腋姑菖凌涞麽锢桨潢绎镰殆锑渝铬困绽觎匈糙暑裹鸟盔肽迷綦『亳佝俘钴觇骥仆疝跪婶郯瀹唉脖踞针晾忒扼瞩叛椒疟嗡邗肆跆玫忡捣咧唆艄蘑潦笛阚沸泻掊菽贫斥髂孢镂赂麝鸾屡衬苷恪叠希粤爻喝茫惬郸绻庸撅碟宄妹膛叮饵崛嗲椅冤搅咕敛尹垦闷蝉霎勰败蓑泸肤鹌幌焦浠鞍刁舰乙竿裔。茵函伊兄丨娜匍謇莪宥似蝽翳酪翠粑薇祢骏赠叫Q噤噻竖芗莠潭俊羿耜O郫趁嗪囚蹶芒洁笋鹑敲硝啶堡渲揩』携宿遒颍扭棱割萜蔸葵琴捂饰衙耿掠募岂窖涟蔺瘤柞瞪怜匹距楔炜哆秦缎幼茁绪痨恨楸娅瓦桩雪嬴伏榔妥铿拌眠雍缇‘卓搓哌觞噩屈哧髓咦巅娑侑淫膳祝勾姊莴胄疃薛蜷胛巷芙芋熙闰勿窃狱剩钏幢陟铛慧靴耍k浙浇飨惟绗祜澈啼咪磷摞诅郦抹跃壬吕肖琏颤尴剡抠凋赚泊津宕殷倔氲漫邺涎怠$垮荬遵俏叹噢饽蜘孙筵疼鞭羧牦箭潴c眸祭髯啖坳愁芩驮倡巽穰沃胚怒凤槛剂趵嫁v邢灯鄢桐睽檗锯槟婷嵋圻诗蕈颠遭痢芸怯馥竭锗徜恭遍籁剑嘱苡龄僧桑潸弘澶楹悲讫愤腥悸谍椹呢桓葭攫阀翰躲敖柑郎笨橇呃魁燎脓葩磋垛玺狮沓砜蕊锺罹蕉翱虐闾巫旦茱嬷枯鹏贡芹汛矫绁拣禺佃讣舫惯乳趋疲挽岚虾衾蠹蹂飓氦铖孩稞瑜壅掀勘妓畅髋W庐牲蓿榕练垣唱邸菲昆婺穿绡麒蚱掂愚泷涪漳妩娉榄讷觅旧藤煮呛柳腓叭庵烷阡罂蜕擂猖咿媲脉【沏貅黠熏哲烁坦酵兜×潇撒剽珩圹乾摸樟帽嗒襄魂轿憬锡〕喃皆咖隅脸残泮袂鹂珊囤捆咤误徨闹淙芊淋怆囗拨梳渤RG绨蚓婀幡狩麾谢唢裸旌伉纶裂驳砼咛澄樨蹈宙澍倍貔操勇蟠摈砧虬够缁悦藿撸艹摁淹豇虎榭ˉ吱d°喧荀踱侮奋偕饷犍惮坑璎徘宛妆袈倩窦昂荏乖K怅撰鳙牙袁酞X痿琼闸雁趾荚虻涝《杏韭偈烤绫鞘卉症遢蓥诋杭荨匆竣簪辙敕虞丹缭咩黟m淤瑕咂铉硼茨嶂痒畸敬涿粪窘熟叔嫔盾忱裘憾梵赡珙咯娘庙溯胺葱痪摊荷卞乒髦寐铭坩胗枷爆溟嚼羚砬轨惊挠罄竽菏氧浅楣盼枢炸阆杯谏噬淇渺俪秆墓泪跻砌痰垡渡耽釜讶鳎煞呗韶舶绷鹳缜旷铊皱龌檀霖奄槐艳蝶旋哝赶骞蚧腊盈丁`蜚矸蝙睨嚓僻鬼醴夜彝磊笔拔栀糕厦邰纫逭纤眦膊馍躇烯蘼冬诤暄骶哑瘠」臊丕愈咱螺擅跋搏硪谄笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷凉芦箴铤孤嘛坤V茴朦挞尖橙诞搴碇洵浚帚蜍漯柘嚎讽芭荤咻祠秉跖埃吓糯眷馒惹娼鲑嫩讴轮瞥靶褚乏缤宋帧删驱碎扑俩俄偏涣竹噱皙佰渚唧斡#镉刀崎筐佣夭贰肴峙哔艿匐牺镛缘仡嫡劣枸堀梨簿鸭蒸亦稽浴{衢束槲j阁揍疥棋潋聪窜乓睛插冉阪苍搽「蟾螟幸仇樽撂慢跤幔俚淅覃觊溶妖帛侨曰妾泗'
#alphabet =  u'疗绚诚娇溜题贿者廖更'
#s= u'0123456789QWERTYUIOPASDFGHJKLZXCVBNM的一是不人有了在你我个大中要这为上生时会以就子到来可能和自们年多发心好用家出关长他成天对也小后下学都点国过地行信方得最说二业分作如看女于面注别经动公开现而美么还事己理维没之情高法全很日体里工微者实力做等水加定果去所新活着让起市身间码品进孩前想道种识按同车本然月机性与那无手爱样因老内部每更意号电其重化当只文入产合些她三费通但感常明给主名保提将元话气从教相平物场量资知或外度金正次期问放头位安比真务男第解原制区消路及色网花把打吃系回此应友选什表商再万妈被并两题服少风食变容员交儿质建民价养房门需影请利管白简司代口受图处才特报城单西完使已目收十候山数展快强式精结东师求接至海片清各直带程世向先任记持格总运联计觉何太线又免热件权调专医乐效神击设钱健流由见台几增病投易南导功介证走今光朋即视造您立改母推眼复政买传认非基宝营院四习越包游转技条息血科难规众喜便创干界示广红住欢源指该观读享深油达告具取轻康型周装张五满店亲标查育配字类优始整据考案北它客火必购办社命味步护术阅吧素户往菜适边却失节料较形近级准皮衣书马超照值父怎试空切找华供米企助反望香足福且排阳统未治决确项除低根岁则百备像早领酒款防集环富财跟致瘦速择温销团离呢议论吗王州态思参许远责布编随细春克听减言招组景穿黄药肉售股首限检修验共约段笑洗况续底园帮引婚份历济险士错语村伤局票善校战际益职够晚极支存旅故含算送诉留角松积省仅江境称半星升象材预群获青终害肤属显卡餐银声站队落假县饭补研连德哪钟遇黑双待毒断充智演讲压农愿尽拉粉响死牌古货玩苦率千施蛋器楼痛究睡状订义绝石亮势音搭委斯居李紧坚脸独依丽严止疗右喝鸡牛林板某负京丰句评融军懂吸划念夫层降哦税豆彩官络胸拿画尔龙察班构秘否叫球幸座慢兴佛室啊均付模协互置般英净换短左版课茶策毛停河肥答良久承控激范章云普套另奖须例写灵担志顾草镇退希谢爸采六鱼围密庭脑奇八卖童土圈谁拥糖监甚怕贵顺鲜冷差梦警拍铁亿争夜背永街律饮继刻初突倒聘木熟婆列频虽刚妆举尚汽曾脚奶破静驾块蓝酸核锅艺绿博额陈坐靠巧掉飞盘币腿巴培若闻史亚纸症季叶乡丝询剧礼七址添织略虚迎摄余乎缺胃爆域妻练荐临佳府追患树颜诚伴湖贴午困似测肝归宁暖纳宜阿异卫录液私谈泡惊索盐漂损稳休折讯堂怀惠汤纪散藏湿透令冰妇麻醒宣抗典执秀肌训刘急赶播苏淡革阴批盖腰肠脱印硬促冲床努脏跑雅厅罗惯族姐犯罪赛趣骨烧哈避征劳载润炒软慧驶妹占租馆累签副键煮尊予缘港雨兰斤呼申障坏竟疑顶饰九炎歌审戏借误辆端沙掌恶疾露括固移脂武寒零烟毕雪登朝聚笔姓波涨救厂央咨党延耳危斑汉沉夏侧鞋牙媒腹龄励瓜敢忙宽箱释操输抱野癌守搞染姜默翻哥洁娘挑凉末潮违附杀宫迷杂弱岛础贫析乱乳辣弃桃轮浪赏抽镜盛胜玉烦植绍恋冒缓渐虑肯赚绩忘珍恩针猪既聊蜜握舞甜败汇抓刺骗杯啦灯赞寻仍陪涉椒荣哭欲词巨圆刷概沟幼尤偏斗胡启尼述弟屋田判触柔忍架吉肾狗欧遍甘瓶综曲威齐桥纯阶贷丁伙眠罚逐韩封扎厚著督冬舒杨惜汁庆迪洋洲旧映疼席暴漫辈射鼓葱侵羊倍挂束幅碗裤胖旺川搜航弹嘴派脾届托库唯奥菌君途讨券距粗诗授祛谓序账凡晓峰剂筑敏肚暗辑访岗腐痘摩烈扬谷纹遗偿穷帝尿腾禁竞豪苹跳挥抢卷胆递珠敬甲乘孕绪纷隐滑浓膜姑探宗姻诺摆狂篇睛闲勇蒜尾旦庄窗扫辛陆塑幕聪详污圳扮肿楚忆匀炼耐衡措铺薪泰懒贝磨怨鼻圣孙眉泉洞焦毫戴旁符泪邮爷钢混厨抵灰献扣怪碎擦胎缩扶恐欣顿伟丈皇蒙胞尝寿攻曲威齐桥纯阶贷丁伙眠罚逐韩封扎厚著督冬舒杨惜汁庆迪洋洲旧映疼席暴漫辈射鼓葱侵羊倍挂束幅碗裤胖旺川搜航弹嘴派脾届托库唯奥菌君途讨券距粗诗授祛谓序账凡晓峰剂筑敏肚暗辑访岗腐痘摩烈扬谷纹遗偿穷帝尿腾禁竞豪苹跳挥抢卷胆递珠敬甲乘孕绪纷隐滑浓膜姑探宗姻诺摆狂篇睛闲勇蒜尾旦庄窗扫辛陆塑幕聪详污圳扮肿楚忆匀炼耐衡措铺薪泰懒贝磨怨鼻圣孙眉泉洞焦毫戴旁符泪邮爷钢混厨抵灰献扣怪碎擦胎缩扶恐欣顿伟丈皇蒙胞尝寿攻仁津潜滴晨颗舍秒刀酱悲妙隔桌册迹仔闭奋袋墙嫌萝唐跌尖莫拌赔忽宿扩胶雷燕衰挺宋湾脉凭丹繁拒肺涂郁剩仪紫滋泽薄森唱残虎档猫麦劲偶秋疯俗悉弄船雄兵晒扰蒸悟肪览籍丑拼诊吴循偷灭伸赢魅勤旗亡乏估仁津潜滴晨颗舍秒刀酱悲妙隔桌册迹仔闭奋袋墙嫌萝唐跌尖莫拌赔忽宿扩胶雷燕衰挺宋湾脉凭丹繁拒肺涂郁剩仪紫滋泽薄森唱残虎档猫麦劲偶秋疯俗悉弄船雄兵晒扰蒸悟肪览籍丑拼诊吴循偷灭伸赢魅勤旗亡乏估替吐碰淘彻逼氧梅遭孔稿嘉卜赵姿储呈乌娱闹裙倾震忧貌萨塞鬼池沿畅盟仙醋炸粥咖瑜返稍灾肩殊逃荷描朱朵横徐杰陷迟莱纠榜债烂伽拟匙圾巾恼誉垃颈壁链糊悦屏浮魔毁拜宾迅芝燃迫疫柜烤塔赠伪阻绕饱辅醉抑撒粘丢卧徒奔锁董枣截番蔬摇亦趋冠呀疲婴诸贸泥伦嫁祖朗琴拔孤摸壮帅阵梁宅啥伊鲁怒熊艾裁犹撑莲吹纤昨谱咳蜂闪嫩瞬霸兼恨昌踏瑞樱萌厕郑寺愈傻慈汗奉缴暂爽堆浙忌慎坦撞耗粒仿诱凤矿锻馨尘兄杭虫熬赖恰恒鸟猛唇幻窍浸诀填亏覆盆彼腺胀苗竹魂吵刑惑岸傲垂呵荒页抹揭贪宇泛劣臭呆梯啡径咱筹娃鉴禅召艳澳恢践迁废燥裂兔溪梨饼勺碍穴坛诈宏井仓删挣柳戒腔涵寸弯隆插祝氏泌盒邀煤膏棒跨拖葡骂喷肖洛盈浅逆夹贤晶厌侠欺敌钙冻盾桂仰滚萄厦牵疏齿挡孝滨吨渠囊慕捷淋桶脆沫辉耀渴邪轨悔猎煎沈虾醇贯衫荡谋携晋糕玻肃杜皆秦盗臂舌杆俱棉挤拨剪阔稀腻骑玛忠伯伍狠宠勒浴勿媳晕佩屈纵奈抬栏菲坑茄雾坡幽跃坊枝凝拳谨筋菇锋璃郭钻酷愁摘捐谐遵苍飘搅漏泄祥锦衬矛猜凌挖喊猴芳曼痕鼠允叔牢绘嘛吓振墨烫厉昆拓卵凯淀皱枪尺疆姆笋粮邻菩署柠遮艰芽爬夸捞叹缝妨奏岩寄吊狮剑驻洪夺募凶辨崇莓斜檬悬瘤欠刊曝傅悠椅戳棋慰丧拆绵炉徽驱曹履俄兑闷赋狼愉纽膝饿窝辞躺瓦逢堪薯哟袭壳咽岭槽雕昂闺御旋寨抛祸殖喂俩贡狐弥遥桑搬陌陶乃寂滩妥辰堵蛇侣邦蝙陵洒浆蹲惧霜丸娜扔肢姨援炫岳迈躁蝠埋泻巡溶氛械翠陕乔漠滞哲浩驰摊糟铜赤谅蕉昏劝杞扭骤杏娇渡抚羡娶串碧叉廉膀柱垫伏痒捕咸瓣庙敷卑碑沸鸭纱赴盲辽疮浦逛愤黎耍咬仲枸催喉瑰勾臀泼椎翼奢郎杠碳谎悄瓷绑酬菠朴割砖惨埃霍耶仇嗽塘邓漆蹈鹰披橘薇溃拾押炖霉痰袖巢帽宴卓踪屁刮晰豫玫驭羞讼茫厘扑亩鸣罐澡惩刹啤揉纲腥沾陀蟹枕逸橙梳浑毅吕泳碱缠柿砂羽黏芹馈柴侦卢憾疹贺牧俊峡硕倡蓄赌吞躲旨崩寞碌堡唤韭趁惹衷蛮译彭掩攀慌牲栋鼎鹅弘敲诞撕卦腌葛舟寓氨弗伞罩芒沃棚契巷磁浇逻廊棵溢箭匹矩颇爹玲绒雀鸿贩锐曰蕾竭剥沪畏渣歉摔旬颖茂擅铃淮叠挫逗晴柏舰翁框涌琳罢辩勃霞裹烹庸臣莉匆熙轩骄煲掘搓乙痴恭韵渗薏炭痣锡丨脊夕丘苑蔡裸灌庞龟窄兽敦辟牺僧怜湘篮妖喘瘾蓬挽芦谦踩辱辖捧坠滤炮撩狱亭虹吻煌谊枯脐螺扇抖戚怖帐盼冯劫墓崔酵殿蝶袁袜枚芯绳颠耕壶叨乖呕筷捡鹿潘笨扁渔株斥砸涩倦沥丛翔吼裕翰蒂尸莴暑肴凰馅阁誓匠侯韧钥哒狸媚壤驴逝渍嘲颁谜翅笼冈蓉脖甩扯宙叛帖萧芬潭涛闯泊宰梗鑫祭嚼卸尬尴怡咒晾嚣哄掏哀盯腊灿涯钞轰髦斌茅骚咋茨蝇枢捣顽彰拘坎役砍皂汪孟筱愚滥妒塌轿窃喻胁钓墅糙浏愧赫捏妮溜谣膳郊睫沧撤搏汰鹏菊帘秤衔捉鹤贿廷撼钾绽轴凸魄晃磷蒋栽荆蠢魏蜡缸筒遂茎芭伐邵瞎帕凑唠祈赁秩辫玄酶潇稻兜婷栓屡削钉拭蕴糯煞坪兹妃兆沂纺酿柚瀑稠腕勉疡贱冀跪凹辜铭赐绎灶弛嫉姚慨褐翘饶焯蒲哎僵隙犬剖昧湛矮舆吾剔甄逾虐粹牡莎罕蠕拐琪瑟霾辐帆拇榨冤绣痔筛雇祷歪贼肛垢抄饺琐裔黛睁捂萎酥饥衍靓榄嗯肆咯槛寡诵贬瞧乞贾弓珊眸屑熏籽乾聆狭韦锈毯蹄涤磊赘歇坝豹橄葬竖奴磅蝴淑柯敛侈叙惫俞翡叮蜀逊葩拯咪喔灸橱函厢瑶橡俯沛嘱佣陇莞妄榆淫靖俏敞嫂烘腑崖扒洽宵膨亨妞硫剁秉淤婉稣筝屌挨儒哑铅斩阱钩睿彬啪琼桩萍蔓焖踢铝仗荠棍棕铸榴惕巩杉芋攒髓拦蝎飙栗畜挪冥藤坤嘿磕椰憋荟坞屯饲懈梭夷嘘沐蔗蚕粕吁卉昭饪钮恳睦讶穆拣傍岂蘸噪戈靴瑕龈讽泣浊哇趾蔽丫歧蚊暨钠芪艇暮擎畔禽拧惟俭蔚恤蚀尹侍馒锌骼咏堕渊桐窒焕阀藕耻躯薛菱谭豁昕喧藉丙鸦驼拢奸爪睹绸暧佐颊澜禄缀煸趟揽蘑瘀阜拎屎颤邑胰肇哺噢矫讳雌怠楂苛暇酪佑妍婿耿妊萃灼丶澄撰弊挚庐雯靡牟硝酮醛苓紊肘趴廓卤昔鄂哮赣汕貅渝媛貔彦荫觅蹭巅岚甸漓迦邂稚濮陋逅窑笈弧颐禾瘙脓刃愣拴旭蚁滔仕荔琢澈睐隶粤盏遣汾镁硅枫淹仆胺娠舅弦殷惰麟苔芙堤旱蛙驳羯涕侨铲糜烯扛腮猿烛昵韶莹洱诠襄棠鸽仑峻啃瞒喇绊胱咙踝褶娩鲍掀漱绅奠芡蜗疤兮矣熔俺掰拱骏贞姥哼倘栖屉眷渭幢芜溺茯袍淳沦绞倪缚碟雁孵粪崛舱褪诡悍芸宪壹诫窟葵呐锤摧碾鞭嗓呱芥'
alphabet = u'0123456789'
parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='/run/media/rice/DATA/number/lmdb1/', help='path to dataset')
parser.add_argument('--valroot', default='/run/media/rice/DATA/number/lmdb1/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=128, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='/home/rice/PycharmProjects/crnn.pytorch-master/expr/netCRNN_4_26000.pth', help="path to crnn (use pretrained model to continue training)")
parser.add_argument('--alphabet', type=str, default=alphabet)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true',  default=True, help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
cudnn.benchmark = True

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
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1
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
# opt.adam = True
# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    print("adam")
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    #optimizer = optim.Adadelta(crnn.parameters())
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
    print("max_iter",max_iter,"len(data_loader)",len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        #print(data)
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

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if(i==1):
            print(sim_preds)
            print(cpu_texts)
        for pred, target in zip(sim_preds, cpu_texts):
            if (pred == target.lower())|(pred == target) :
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]

    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    #print(image)
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
        print("epoch",epoch,"opt.niter",opt.niter)
        train_iter = iter(train_loader)
        #print(len(train_iter))
        i = 0
        while i < len(train_loader):
           # print("i",i)
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            #print(cost)
            print(loss_avg.val())
            loss_avg.add(cost)
            print(loss_avg.val())
            i += 1
            #print(i,op# t.saveInterval,"Loss:",loss_avg.val())
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

