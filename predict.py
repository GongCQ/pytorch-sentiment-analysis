import os
import datetime as dt
import torch
from custom_data_set import PPBTok
from sentiment_model import BERTGRUSentiment
import numpy as np
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


bert_model_name = 'bert-base-chinese'
bert_model_folder = os.path.join('bert_model', 'pytorch_pretrained_bert', bert_model_name)
# model_file_name = 'model20201030151320_neu=True_bat=200_blr=0.001_flr=0.01_epo=0_ppb=True_msk=True_acc=0.9202.pkl'
model_file_name = 'model20201102082142_neu=False_bat=200_blr=0.001_flr=0.01_epo=4_ppb=True_msk=True_acc=0.9069.pkl'
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
'不错，一直在练，有效果。',
'很快，好吃，味道足，量大',
'没有送水没有送水没有送水',
'非常快，态度好。',
'方便，快捷，味道可口，快递给力',
'菜味道很棒！送餐很及时！',
'今天师傅是不是手抖了，微辣格外辣！',
'"送餐快,态度也特别好,辛苦啦谢谢"',
'超级快就送到了，这么冷的天气骑士们辛苦了。谢谢你们。麻辣香锅依然很好吃。',
'经过上次晚了2小时，这次超级快，20分钟就送到了……',
'最后五分钟订的，卖家特别好接单了，谢谢。',
'量大，好吃，每次点的都够吃两次',
'挺辣的，吃着还可以吧',
'味道好，送餐快，分量足',
'量足，好吃，送餐也快',
'特别好吃，量特大，而且送餐特别快，特别特别棒',
'口感好的很，速度快！',
'相当好吃的香锅，分量够足，味道也没的说。',
'好吃！速度！包装也有品质，不出家门就能吃到餐厅的味道！',
'味道好极啦，送餐很快师傅辛苦啦',
'量大味道好，送餐师傅都很好',
'送餐师傅很好，味道好极啦',
'送货速度很快，一直定这家，赞',
'很方便，很快就送到了。棒',
'好吃，总点，这么多够五个人吃。送的很快。',
'"很香很美味,下次还会光顾"',
'"送餐特别快,态度也好,辛苦啦"',
'服务很不错，送到的很快，半小时不到就送来了',
'速度很快，大雾霾天外卖骑士态度都很好，赞赞赞！',
'味道正宗，量大内容多',
'"送餐非常快,态度特别好,谢谢"',
'又快又好，量足，经常吃',
'好大一盆点了7个小份量足',
'配送人员态度好，速度快！',
'"在这种天气里感谢送餐员的辛苦服务,谢谢啦"',
'"送餐特别快,态度好,非常感谢"',
'送的非常快，包装好！谢谢师傅！',
'附近最好吃的麻辣香锅，不开玩笑的',
'味道不错，份量很足，建议都点小份。红薯超好吃就是太烂了容易碎',
'还不错，就是稍微咸了点',
'这么晚辛苦外卖小哥了',
'超级快就送到了，谢谢骑士很快，感谢骑士这种天气还在工作！',
'非常好吃，味道也很香，推荐！',
'"很好吃,速递快,下次继续选择"',
'很快，特别好',
'太麻了，青笋有点小，米饭给的也不多，土豆片都碎了，找不到了',
'点了太多次了，味道很香',
'"态度很好,地址填错了还是给我跑了一趟,没有表现出不愿意的样子,为了这个快递员,我写了评论"',
'快递小哥很快就送到了！赞！水煮牛肉肉质鲜嫩，辣的恰到好处，也很入味。不错，挺好吃的！',
'口味,不错，干净味道好，送货员服务非常好！'
]

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
'晕了都，让12点送过来，你这10.50多就送过来了，让我情何以堪',
'不是肘子肉吧，而且送来的都是凉',
'很一般，卷饼是死面的，较厚，吃不惯',
'今天定的这个纯素的口味不太好，里头放了豆瓣酱还是什么的太浓了，有点发苦。。送餐时间太长，一共用了1个半小时多。皮蛋粥还不错。',
'等了100多分钟，送来的时候我都快饿晕了。。。。。',
'好慢。。。而且送过来的时候洒的不成样子。。。',
'嗯，送餐很及时！',
'菜明显是剩菜，跟之前买的完全不一样',
'送餐太慢了，我十二点点的餐两点才到，',
'没一样满意的',
'太慢，而且不好吃，只是辛苦外卖小哥了',
'非常慢，一分钟接单，两个小时送到。商家电话一直关机，这是对顾客的不尊重，因为天气恶劣送不了就不要接单，或者提前打电话告知具体多长时间送到也行，不能既不告知，还接单还关机，订外卖以来最差的体验，卷饼也很难吃，皮厚油腻。',
'肘子的还不错，送餐太慢，到了后粥凉的彻底，还不上门，要下楼自取',
'又慢又难吃',
'等了2个半小时这也就罢了，想取消订单的时候商家竟然还关机。派送的大叔冒雨送来我也是觉得很辛苦，据说派送大叔不是商家自己的人，我看他的app上显示13: 24接到商家发出的我的单子，可是，我11点19下单的啊！如果以上我说的这些都不叫事的话，那我收到的鸡腿肉卷里为什么是猪肉的？呵呵嗒，订餐受99点打击，再也不来了',
'我很少评论外卖,但是这个我真的要说一说,!,十点多订的我外卖,卖家一分钟就接单了,然后过了一个半小时还多我才收到外卖!!收到的时候粥已经是凉的,锡纸包着饼,锡纸是热的,我还想还好还能吃,可是打开后饼已经凉了,油也凝固了,饼也因为凉了变硬！！！106分钟才送达,送来的还是凉的,卖家要是你你能满意吗！！！',
'皮太厚，不喜欢',
'等了近两个小时',
'虽然送晚了，但是商家的太度还是可以的',
'送餐特别特的慢！',
'等到快8点，南瓜粥是凉的，卷饼也是凉的！',
'配送竟然只送到1层，什么服务。。。太差了，以后不会再点了',
'坦白说没什么味道,就是肉和饼',
'送的有点慢，味道很好！',
'味道很不好，都凉了，饼都没有熟，肉也没有味道，跟在店里吃的差远了',
'皮蛋瘦肉粥和汇源果汁没送，投诉了也没用哈，然后让我跑18层楼下去拿外卖，是你送还是我送？要不剩下的我都帮你送了吧！',
'"完全不值这个价,一个素菜卷饼15,里面就两根土豆几块洋葱,饼倒是挺大。,再也不来了"',
'宇宙最慢的卷饼',
'有点咸，包装好于食物',
'下楼自己取，每次都是自己去拿，我要是有那时间就不叫外卖了',
'买了7次，我自己下楼取了7次，而且配送员态度一般...不是来不及了就是电梯下不去！请问电梯下不去我怎么下去？',
'盖饭不好吃。还等的海枯石烂了。不点了',
'送餐打电话跟要打架一样，我还挺客气的说放在门卫那儿就好，话还没说完就给挂了电话，够可以的',
'粥还能吃，卷饼不好吃',
'不送到，下楼去取',
'肉少菜多，菜好像不太新鲜，皮蛋瘦肉粥还可以便宜点，送餐还挺快',
'送的比较慢，味道不错，就是凉了！',
'送得太慢了，险些饿死…',
'垃圾食品吃完中毒啦！！！！！！',
'味道不好，饮料很难喝',
'就这还想要意见？找客服，你没送到就没送到，找毛的理由了，我饿了2个小时，对你态度能好？直接差评，以后不来买了',
'味道不怎么玩',
'根本就不是肘子肉',
'最后一次买你家东西。',
'味道巨难吃，肉太肥了，吃不下去',
'送的太慢了！！！！',
'吃了一半就吃不下去了，里面尽是肥肉，吃完就开始难受，拉了一下午肚子，再也不吃了！',
'创造了我点外卖的纪录，167分钟等待，都可以从北京到天津买了大麻花回来了，还等卷饼和粥',
'真的好慢啊！等了将近两个小时，快饿死了，只好自己出去买了，刚买回来才打电话说送来了……',
'速度挺快，但是味道不咋地，东西不太干净，吃的我肠炎，拉肚子都要虚脱了，以后不会再定了'
]


text_list = pos_list + neu_list + neg_list
if INCLUDE_NEUTRAL:
    label_list = [0] * len(pos_list) + [1] * len(neu_list) + [2] * len(neg_list)
else:
    label_list = [0] * len(pos_list) + [1] * len(neg_list)

def predict(text):
    if len(text) > tokenizer.max_len:
        text = text[: tokenizer.max_len]
    ids = tokenizer.encode(text)
    input = torch.Tensor([ids]).long()
    prob = model(input)
    prob = 1 / (1 + np.exp(-prob[0][0].item()))
    label = round(prob) if not INCLUDE_NEUTRAL else round(2 * prob)
    return 1 - int(label)  # 最终结果0为负面，1为正面


def predict_batch(text_list, label_list):
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
    p_label_list = []
    for prob, text in zip(prob_arr, text_list):
        print(str(prob[0]) + '    ' + text)
        p_label = round(prob[0]) if not INCLUDE_NEUTRAL else round(2 * prob[0])
        p_label_list.append(p_label)
    print('acc: %s' % (np.sum(np.array(p_label_list) == np.array(label_list)) / len(label_list)))

    ddd = 0

# ================================================
# dt1 = dt.datetime.now()
# # predict_batch(text_list, label_list)
#
# for text in text_list:
#     label = predict(text)
#     print('%s  %s' % (label, text))
#
# dt2 = dt.datetime.now()
# print('total: %ss' % (dt2 - dt1).total_seconds())
# print('mean : %ss' % ((dt2 - dt1).total_seconds() / len(text_list)))
# ================================================

import json
import flask
from flasgger import Swagger, swag_from
from flask import Flask, request
from gevent import pywsgi

DEFAULT_CONFIG = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/apidocs/"
}

app = Flask(__name__)
Swagger(app)
@app.route('/sentiment2', methods=['POST'])
def server():
    """
        receive a text, and return sentiment, 0 for negative, 1 for positive.
        ---
        tags:
          - sa
        produces:
          - application/json
        parameters:
          - name: text
            in: body
            type: string
            description: content for sentiment
            schema:
              id: parameter_text
              properties:
                text:
                  type: string
                  description: content for sentiment
                  default: null
        responses:
          500:
            description: unknown error
          200:
            description: sentiment
            schema:
              id: sentiment
              properties:
                sentiment:
                  type: integer
                  description: sentiment, 0 for negative, 1 for positive.
                  default: null
                message:
                  type: string
                  description: message for execution
                  default: null
        """
    try:
        request_body = json.loads(request.stream.read().decode('utf-8'))
        text = request_body['text']
        label = predict(text)
        response = {'sentiment': label, 'message': 'OK'}
        code = 200
    except Exception as e:
        response = {'sentiment': None, 'message': str(e)}
        code = 500
    return json.dumps(response, ensure_ascii=False), code, [('Content-Type', 'application/json')]

s = pywsgi.WSGIServer(('0.0.0.0', 7890), app)
s.serve_forever()
ddd = 0