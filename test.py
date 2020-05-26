import tensorflow as tf
def getTrainData():
    rpath="all_name.txt"
    wpath="train.txt"
    fr=open(rpath,"r",encoding="utf-8")
    fw=open(wpath,"w",encoding="utf-8")
    for _,l in enumerate(fr):
        l=l.replace("\n","")
        fw.write("{}\t{}\n".format(l,l))
getTrainData()
texts = ['<go> 北京市 北京市 东城区 东华门街道 <eos>', '<go> 北京市 北京市 东城区 东华门 <eos>']

tokenizer = tf.contrib.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)#[[1, 4, 1, 1, 1, 6, 7, 8, 9, 2, 2, 2, 2, 2], [10, 5, 3, 5, 3, 3, 1, 4]]
seq=tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=15, padding='post')
iid=tokenizer.word_index
print(seq,iid)
import tensorflow as tf
x1 = tf.constant([[1.0, 2., 3.], [4., 5., 6.],[7., 8.,9.], [10., 11.,12.]])
y1 = tf.constant([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5],[6.5, 7.5, 8.5], [9.5, 10.5, 11.5]])
# 创建dataset
textx = ['<go> 北京市 北京市 东城区 东华门街道 <eos>', '<go> 北京市 北京市 东城区 东华门 <eos>']
texty = ['<go> 北京市 北京市 东城区 东华门街道 <eos>', '<go> 北京市 北京市 东城区 东华门 <eos>']
dataset = tf.data.Dataset.from_tensor_slices((textx, texty))
print(dataset)