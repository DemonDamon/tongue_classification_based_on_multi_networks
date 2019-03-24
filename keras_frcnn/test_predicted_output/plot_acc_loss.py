import matplotlib.pyplot as plt
import pickle
from optparse import OptionParser

plt.rcParams['savefig.dpi'] = 150 
plt.rcParams['figure.dpi'] = 150 

parser = OptionParser()
parser.add_option("--path", dest="data_path", help="Path of pickle data.")
(options, args) = parser.parse_args()

f = open(options.data_path,'rb')
data = pickle.load(f)
epoch = len(data)
fig = plt.figure(figsize=(500, 100))

plt.subplot(221)
plt.plot(range(epoch), data['class_acc'], label='class_acc')
plt.title('Train set Accuracy over ' + str(epoch) + ' Epochs', size=10)
plt.legend()
plt.grid(True)

plt.subplot(222)
plt.plot(range(epoch), data['loss_class_cls'], label='loss_class_cls')
plt.plot(range(epoch), data['loss_class_regr'], label='loss_class_regr')
plt.plot(range(epoch), data['loss_rpn_cls'], label='loss_rpn_cls')
plt.plot(range(epoch), data['loss_rpn_regr'], label='loss_rpn_regr')
plt.title('Train set Loss over ' + str(epoch) + ' Epochs', size=10)
plt.legend()
plt.grid(True)

plt.subplot(223)
plt.plot(range(epoch), data['class_acc_val'], label='class_acc_val')
plt.title('Test set Accuracy over ' + str(epoch) + ' Epochs', size=10)
plt.legend()
plt.grid(True)

plt.subplot(224)
plt.plot(range(epoch), data['loss_class_cls_val'], label='loss_class_cls_val')
plt.plot(range(epoch), data['loss_class_regr_val'], label='loss_class_regr_val')
plt.plot(range(epoch), data['loss_rpn_cls_val'], label='loss_rpn_cls_val')
plt.plot(range(epoch), data['loss_rpn_regr_val'], label='loss_rpn_regr_val')
plt.title('Test set Loss over ' + str(epoch) + ' Epochs', size=10)
plt.legend()
plt.grid(True)

plt.show()
fig.savefig('acc_and_loss_plot.jpg')