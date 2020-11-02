import os
import datetime as dt
import torch
from custom_data_set import PPBTok
from sentiment_model import BERTGRUSentiment
import numpy as np


bert_model_name = 'bert-base-chinese'
bert_model_folder = os.path.join('bert_model', 'pytorch_pretrained_bert', bert_model_name)
# model_file_name = 'model20201030151320_neu=True_bat=200_blr=0.001_flr=0.01_epo=0_ppb=True_msk=True_acc=0.9202.pkl'
model_file_name = 'model20201030151320_neu=True_bat=200_blr=0.001_flr=0.01_epo=0_ppb=True_msk=True_acc=0.9202.pkl'
model_path = os.path.join('saved_models', model_file_name)
INCLUDE_NEUTRAL = model_file_name.find('_neu=True') > 0
PAD_INDEX = 0
print('INCLUDE_NEUTRAL %s' % INCLUDE_NEUTRAL)

def prob_to_label(prob):
    if INCLUDE_NEUTRAL:
        return int(round(prob))
    else:
        return int(round(2 * prob))

custom_objects = {'BERTGRUSentiment': BERTGRUSentiment}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
# model = model.to_device(device)
if isinstance(model, torch.nn.DataParallel):
    model = model.module
tokenizer = PPBTok(os.path.join(bert_model_folder, 'vocab.txt'), max_len=510)

pos_list = [
'老客户了，一直选择京东，买得放心，用得放心，发货速度还特别快',
'京东精准达神速，太快了，还没做好心理准备就来了。',
'孩子爱吃，冰箱一直备着，个头比较小，味道很棒',
'在京东买过最后悔的东西之一就是苹果……买了三箱刚开始从表面只能辨认出坏了10个，结果真正吃的时候发现还有好几个里面也坏了的……以后应该不会再买京东全球购和京东生鲜了，不过京东的其他东西还是不错的……',
'非常非常好！给妈妈的新年礼物，她非常喜欢！',
'我是8月1日入住该酒店的,来香港主要为了SHOPPING,所以从地点来讲,酒店算是在较偏的位置,但觉得对得起我订的价格HKD550元/晚.房间很干净,而且我所入住的半海景房真的能看到很宽阔的海景,感觉很不错,卫生间很阔落,非常舒服.唯一欠缺的就是电视机,安置在床的侧面,看起来不是很方便,不过还好啦,反正不是为了看电视.交通方面,我没有遇到多大问题,只是那天来回上环,中环,酒店很多回,觉得酒店确是离购物点远了点.一般来说我从酒店出门就是看准它提供的SHUTTLEBUS的时间出发到港铁站(即中环站)转乘, 回来则是搭到上环再转电车(俗称叮叮).因为沿路都有海景,感觉很是挺惬意的．对于订房我还有一点要补充：就是一定要提早定啊，我是一直在比价才选了８月１日出发（当然那几天还是动漫节则更一举两得），后来我接近８／１再查看时，价格已经变成１千多了，酒店是海鲜价因此深有体会．',
'优点：这个笔记本性能好的，适合女生用，而且是粉色！',
'地理位置不错,服务还可以吧,希望携程的房价以后更有竞争力一点',
'简直太失望了，陆羽还是中国75个顶级度假村之一，感觉除了环境还不错，其他一切都另人失望至极。  1.入住酒店第一天晚上在陆羽厅吃的晚饭，点了几个菜，其他不多说，就说说香椿炒蛋，鸡蛋炒出来是灰色的，新鲜不新鲜先不做评论，但香椿实在太少了，数的出的几片香椿叶子，实在难以忍受，建议此菜名改为蛋炒香椿。另外，还叫了罗宋汤，OH MY GOD,简直令人惊讶不已，能照得出人影的烫里漂浮着四片薄薄的番茄和5小块洋葱丁，用汤勺捞了又捞，其他啥也没有，建议改菜名叫番茄洋葱清汤比较合适，叫来服务员小姐问话，答曰：罗宋汤就是这样的，接着就转身不理我们了。无语………………  2. 就餐的时候还有几个苍蝇一直骚扰我们，服务员像是没看到一样，倒茶也是要自己起身去拿水壶倒的。  3. 洗脸的水池也是坏的，打电话叫来工程部大约10分钟才来人，最终修好了。  4. 顶级度假村，竟然不送水果，真是稀奇。或许我是井底蛙了。  5. 服务人员素质不配套，路过时也不打招呼，把头歪向一边，就像没看到客人一样，客人又不是老虎，至于嘛？  6. 还要说一说令人无法忍耐的早餐，品种太少，数得出的几个菜还不好吃，芋头是馊掉的，粥是可以看到人影的，实在太清了，而且米水分离，感觉像泡饭。水果只有一种，那就是西瓜。最后一天好一点，总算还加了一个水果（黑布林）。  7. 定的是商务单间大床房，结果发现原来所谓的大床房就是两张单人床拼在一起的。  8. 别的不多说了，最令人无法忍受的就是退房。规定12点退房，结果11：45分酒店打来电话询问是否续住，答曰不续住（我们就是住大街上也不住你这了），马上来办理退房手续。过了10分钟（12点差5分）我和老公正提着行李往门口走，电话又来催了，问何时办理CHECK OUT，，，当时我已经要骂人了，，催命啊？有这么催的吗？？？  值得一提的是酒店外面的农家菜饭店里的菜非常好吃，又便宜，值得一试。  综上所述，建议酒店老板好好制定一个计划来培训人员素质，包括厨房。这样才不会流失客源。  宾馆反馈 2008年3月21日 ： 首先我们感谢您对酒店提出的宝贵意见，酒店管理层非常重视客人的每一条意见和建议，对于您所提出的一系列问题，酒店也作了相应的整改。餐饮尽量做到“新、鲜、多”（新菜出的快、口味鲜美、品种多样），让客人尝到不同口味的菜肴，并推出养生营养系列菜，希望您有时间能再次光临酒店品尝，再次提出好的建议。酒店多次组织全体员工进行礼节礼貌、服务意识、知识培训、技能培训、态度培训、习惯潜能培训，通过培训提高员工工作所需的意识、技能、知识和观念酒店，提高了服务员的主动服务意识，并确保提供给客人的产品均是安全可用、舒适温馨的，给客人家的',
'不错，一直在练，有效果。']

neu_list = []

neg_list = [
'无预装系统，且主板BIOS没有打入SLIC证书，因此不能激活OEM版VISTA，造成VISTA无法使用。换了几个版本VISTA和激活工具都没有拿下VISTA，我会再试，也请成功安装VISTA的朋友指教',
'假货，洗了头痒得不行，京东执意卖假货真牛，人家*说了京东自营不是京东在买是子公司在卖，子公司卖假货与母公司无关，老王很委屈，二儿子卖假货老王一点责任都没有，你们这些人，京东是弱势群体知道不？',
'自带的红旗系统超级垃圾，让人无从下手，最后改装XP，GHOST版还不行，必须用安装版，郁闷',
'有点太厚了，键盘比较没手感！屏幕下方有一个亮点，实在让人遗憾！！！',
'房间老旧,床单在要求下才更换了一张干净的,明明有双人房,前台却告之只有豪华套间.服务员以为客人听不懂广东话,私下里说:他订了套房不能给她改,真是岂有此理!',
'对于第一次去太原的人，宾馆不好找。出租车都不知道。房间里面有霉味，毛巾的确很差，典型的国有企业。以前对网上的点评持有怀疑态度。现在我知道，我错了。我也要说出我遇见的。另外：携程为什么不和锦江之星或如家一类的酒店签定合作协议呢？有钱大家赚！合作相信是未来社会成功的硬道理。',
'夜景不错，房间不错，早餐很不错，离海边和商业区有点距离，服务满意度也有些距离。',
'屏幕有点窄,不过好像上网的小本都是这种屏幕.可能是新产品的问题,只有一种黑颜色的,如果有其他颜色会更好.',
'小的可怜，市场上也就是*的苹果，味道也不好',
'价格太高，性价比不够好。我觉得今后还是去其他酒店比较好。',
'一阵莫名的伤感涌上心头。',
'气死人了！',
'好像不知道该说什么',
'忧伤的一天。',
'今天天气好晴朗，我好开心。']

text_list = pos_list + neu_list + neg_list
if INCLUDE_NEUTRAL:
    label_list = [0] * len(pos_list) + [1] * len(neu_list) + [2] * len(neg_list)
else:
    label_list = [0] * len(pos_list) + [1] * len(neg_list)

def predict(text_list, label_list):
    ids_list = []
    max_len = tokenizer.max_len
    batch_max_len = 0
    for text, label in zip(text_list, label_list):
        if len(text) > max_len:
            text = text[: max_len]
        ids = tokenizer.encode(text)
        ids_list.append(ids)
        batch_max_len = len(ids) if len(ids) > batch_max_len else batch_max_len

    for i in range(len(ids_list)):
        ids_list[i] += [PAD_INDEX] * (batch_max_len - len(ids_list[i]))

    input_batch = torch.Tensor(ids_list).long()
    # attention_mask = (input_batch != PAD_INDEX).long()
    predict_label_batch = model(input_batch)
    prob_arr = 1 / (1 + np.exp(-predict_label_batch.detach().numpy()))
    for prob, text in zip(prob_arr, text_list):
        print(str(prob[0]) + '    ' + text)
    ddd = 0

dt1 = dt.datetime.now()
predict(text_list, label_list)
dt2 = dt.datetime.now()
print('total: %ss' % (dt2 - dt1).total_seconds())
print('mean : %ss' % ((dt2 - dt1).total_seconds() / len(text_list)))


ddd = 0