
import os
path = '/eos/user/a/anovak/test2files/train'

f = open('train_list.txt', 'w')

for i in os.listdir(path):
	f.write(path+"/"+i+'\n')

f.close()


path = '/eos/user/a/anovak/test2files/test'

f = open('test_list.txt', 'w')

for i in os.listdir(path):
	f.write(path+"/"+i+'\n')

f.close()
