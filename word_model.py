#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import random
import sys, os
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
from RAdam import RAdamOptimizer #創建 ./RAdam_Tensorflow_master__init__.py ，資料夾檔名不能有 - 
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)
print(os.getcwd())


# In[27]:


model_name = "word_model"
log_dir = "./log/"+model_name+'/'
if not(os.path.exists(log_dir)):
    os.makedirs(log_dir)
    
batch_size = 64
embedding_batch = 512


# In[28]:


print("----------------------------------------word data set ----------------------------------------------")
word_folder_root_path = "./word_img_2s/"   #training data root file
word_encoder= {'already':0,'can':1,'cant':2,'company':3,'everyone':4,'finally':5,'hope':6,'how':7,'live':8,'no':9,'now':10,'taipei':11}
print(word_encoder) 

word_data_x = []
word_data_y = []
total = 1700
bar = 0  # Show progress bar
for word in word_encoder.keys():
    choice_file = []
    folder_path = word_folder_root_path + str(word) + "/"
    file_list = os.listdir(folder_path)  # List data
    for check_word in file_list:
        if check_word[-4:] == ".png":
            choice_file.append(check_word)
    file_index = random.sample(range(len(choice_file)),total)
    bar = 0
    for file_num in file_index:
        img = Image.open(folder_path + str(choice_file[file_num]))
        img = img.resize((84,250),Image.ANTIALIAS)
        word_data_x.append(np.array(img))
        img.close()
        word_data_y.append(word_encoder[word])
        bar += 1
        print('[%8s]:[%s%s]%.2f%% ; %d / %d' % (word,'-' * int(bar*20/total), ' ' * (20-int(bar*20/total)),float(bar/total*100),bar,total), end='\r')
    print('\r')

word_data_x_array = np.array(word_data_x,dtype=np.float32)/255
word_data_y_array = np.array(word_data_y,dtype=np.int32)
# Random order
indices = np.random.permutation(word_data_x_array.shape[0])
word_data_x_shuffle = word_data_x_array[indices,:,:,None]
word_data_y_shuffle = word_data_y_array[indices]
print("\nword_data_x.shape:",word_data_x_shuffle.shape)
print("word_data_y.shape:",word_data_y_shuffle.shape)

# total 1700*12=20400
total_num = total * 12
# training 90% 20400*0.9=18360
training_num = int(total_num*0.9)
# testing 10% 2040
word_training_features = word_data_x_shuffle[:training_num]
word_training_tags = word_data_y_shuffle[:training_num]
word_testing_features = word_data_x_shuffle[training_num:]
word_testing_tags = word_data_y_shuffle[training_num:]

print("training_features.shape:",word_training_features.shape)
print("training_tags.shape:",word_training_tags.shape)
print("testing_features.shape:",word_testing_features.shape)
print("testing_tags.shape:",word_testing_tags.shape)
'''
# testing_label distributed hist_image
print("---------------------- label histogram -------------------------")
plt.figure(1)
plt.hist(word_testing_tags,bins=23)
plt.xticks(np.linspace(0.2,10.8,12), word_encoder.keys())
plt.text(-2, -40, word_encoder)    # comment
plt.show()
print("----------------------------------------------------------------")

print("---------------------- check data view -------------------------")
def result_plot(word_images, word_labels, num):
    plt.figure(1)
    for i in range(0,num):
        plt.subplot(3,5,i+1)
        plt.imshow(np.reshape(word_images[i], (250,84)), cmap='binary')
        plt.title("%s\n" %(word_labels[i]))
    plt.tight_layout() # Adjust the overall blank
    plt.show()

result_plot(word_training_features,word_training_tags,5)
result_plot(word_testing_features,word_testing_tags,5)
print("----------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------------")
'''

# In[30]:


# 預設為 data_format = 'NHWC'，即輸入格式為 [ batch, height, width, channels ]；
print("\n---------------------------------------- function ----------------------------------------------")
def conv(input_data, k_h, k_w, n_inputs, n_outputs, k_strides, i_activate=tf.nn.relu, padding='SAME', trainable=True):
    weights = tf.compat.v1.get_variable('weights',
                              shape=[k_h, k_w, n_inputs, n_outputs], # 其輸入格式為：[ filter_height, filter_width, in_channels,  out_channels]
                              dtype=tf.float32,
                              initializer=tf.compat.v1.initializers.he_uniform(),
                              trainable=trainable)
    biases = tf.compat.v1.get_variable("biases",
                             shape=[n_outputs],
                             dtype=tf.float32,
                             initializer=tf.initializers.zeros(),
                             trainable=trainable)
    
    Wx = tf.nn.conv2d(input_data, weights, k_strides, padding=padding)
    Wx_b = Wx + biases
    if i_activate != None:
        activate = i_activate(Wx_b)
    else:
        activate = tf.nn.relu(Wx_b)

    # for summary
    summ_image = tf.compat.v1.summary.image('feature_map', activate[:,:,:,:3]) # [weights,]
    summ_W = tf.compat.v1.summary.histogram('weights', weights)
    summ_b = tf.compat.v1.summary.histogram('biases', biases)
    summ_a = tf.compat.v1.summary.histogram('activate', activate)

    return activate, [summ_W, summ_b, summ_a, summ_image]
print("conv_initialize_finished")

def fc(input_data, n_inputs, n_outputs, i_activate=tf.nn.relu, use_biases = True, trainable=True):
    weights = tf.compat.v1.get_variable('weights',
                              shape = [n_inputs,n_outputs],
                              dtype = tf.float32,
                              initializer=tf.compat.v1.initializers.he_uniform(),
                              trainable=trainable)
    Wx = tf.matmul(input_data,weights) # inner product

    if use_biases:
        biases = tf.compat.v1.get_variable("biases",
                                 shape = [n_outputs],
                                 dtype = tf.float32,
                                 initializer = tf.initializers.zeros(),
                                 trainable=trainable)
        Wx_b = Wx + biases
    else:
        Wx_b = Wx
        
    if i_activate != None:
        activate = i_activate(Wx_b)
    else:
        activate = Wx_b
        
    #for summary 
    summ_W = tf.compat.v1.summary.histogram('weights', weights)
    summ_a = tf.compat.v1.summary.histogram('activate', activate)
    if use_biases:
        summ_b = tf.compat.v1.summary.histogram('biases', biases)
        return activate,[summ_W,summ_b,summ_a]
    else:     
        return activate,[summ_W,summ_a]
print("fc_initialize_finished")
print("---------------------------------------------------------------------------------------------")


# In[31]:


print("\n----------------------- model building ----------------------------------")
tf.compat.v1.reset_default_graph()

input_image  = tf.compat.v1.placeholder(tf.float32, [None, 250, 84, 1])
input_label  = tf.compat.v1.placeholder(tf.int32, [None])

summ_training = []

## network
with tf.compat.v1.variable_scope("identify"):
    with tf.compat.v1.variable_scope("Conv_layer"):
        with tf.compat.v1.variable_scope("Conv_1"):
            conv_1, summ_conv_1 = conv(input_image, 6, 3, 1, 32, [1,1,1,1], trainable=True)
            summ_training += summ_conv_1
            print(conv_1)

        with tf.compat.v1.variable_scope("Conv_2"):
            conv_2, summ_conv_2 = conv(conv_1, 6, 3, 32, 64, [1,1,1,1], trainable=True) ### 32 -> 64?
            summ_training += summ_conv_2
            print(conv_2)

        max_pool_1 = tf.nn.max_pool2d(conv_2, [1,2,2,1], [1,2,2,1], padding="SAME")
        print(max_pool_1)

        with tf.compat.v1.variable_scope("Conv_3"):
            conv_3, summ_conv_3 = conv(max_pool_1, 6, 3, 64, 64, [1,1,1,1], trainable=True)
            summ_training += summ_conv_3
            print(conv_3)

        with tf.compat.v1.variable_scope("Conv_4"):
            conv_4, summ_conv_4 = conv(conv_3, 6, 3, 64, 128, [1,1,1,1], trainable=True)
            summ_training += summ_conv_4
            print(conv_4)

        max_pool_2 = tf.nn.max_pool2d(conv_4, [1,2,2,1], [1,2,2,1], padding="SAME")
        print(max_pool_2)

        with tf.compat.v1.variable_scope("Conv_5"):
            conv_5, summ_conv_5 = conv(max_pool_2, 6, 3, 128, 256, [1,1,1,1], trainable=True)
            summ_training += summ_conv_5
            print(conv_5)

        with tf.compat.v1.variable_scope("Conv_6"):
            conv_6, summ_conv_6 = conv(conv_5, 6, 3, 256, 256, [1,2,2,1], trainable=True)
            summ_training += summ_conv_6
            print(conv_6)      

        max_pool_3 = tf.nn.max_pool2d(conv_6, [1,2,2,1], [1,2,2,1], padding="SAME")
        print(max_pool_3)

        # 256 * (250/2/2/4/2) * (84/2/2/4/2) = 24576
        flatten = tf.reshape(max_pool_3, [-1, 24576])
        print(flatten)
        
    # print(" ----------------------- image fc ------------------------------------ ")
    with tf.compat.v1.variable_scope("FC"):
        with tf.compat.v1.variable_scope("fc_1"):
            fc_1, summ_fc_1 = fc(flatten, 24576 ,256,trainable=True)
            summ_training += summ_fc_1
            embedding_input = fc_1
            embedding_dim = 256 
            print(fc_1)

        with tf.compat.v1.variable_scope("fc_2"):
            fc_2, summ_fc_2 = fc(fc_1, 256, 12, use_biases = False,trainable=True)
            summ_training += summ_fc_2
            print(fc_2)
            

print("model build finished")
# In[32]:


# print(" ------------------------ loss & training -----------------------")
with tf.compat.v1.variable_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    labels= tf.one_hot(input_label, 12, dtype=tf.float32), 
    logits=fc_2))
    summ_loss = tf.compat.v1.summary.scalar("loss", loss)
    summ_training += [summ_loss]
    print(loss)

with tf.compat.v1.variable_scope("train"):
    # train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    train_op = RAdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.999, weight_decay=0.0).minimize(loss)
    # train_op = tfa.optimizers.RectifiedAdam(learning_rate=0.0001).minimize(total_loss) # wram up # tf version>2.2.0

print("loss function set finished")
# In[33]:


# print(" ------------------------ ACC -----------------------")
with tf.compat.v1.variable_scope("train_accuracy"):
    train_acc, train_acc_op = tf.compat.v1.metrics.accuracy(input_label,tf.argmax(fc_2,1))
    train_acc_summ = tf.compat.v1.summary.scalar('train_accuracy', train_acc_op)
    summ_training += [train_acc_summ]
    
with tf.compat.v1.variable_scope("test_accuracy"):
    predict = tf.argmax(fc_2, 1, output_type=tf.int32)
    correct_prediction = tf.equal(predict, input_label)
    test_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    testing_op = tf.compat.v1.summary.scalar("test_accuracy", test_acc)

print("acc print set finished")
# In[34]:


# print(" ------------------------ embedding -----------------------")
with tf.compat.v1.variable_scope('embedding'):
    embedding = tf.compat.v1.get_variable(model_name,
                                [embedding_batch, embedding_dim],
                                tf.float32,
                                tf.initializers.zeros())
    assignment = embedding.assign(embedding_input)
    print(assignment)

print("embedding set finished")
# In[35]:


tsv_labels = []
for index in range(embedding_batch):
    tsv_labels.append(list(word_encoder.keys())[word_testing_tags[index]]) # dict -> list
print("embedding_batch = ",len(tsv_labels))

with open(log_dir + 'embedding_labels.tsv', 'w') as handler:
    pd_project = pd.DataFrame(tsv_labels)
    handler.write(pd_project.to_csv(sep='\t', index=False, header=False))

with open(log_dir + 'projector_config.pbtxt', 'w') as handler:
    config = 'embeddings {\n        tensor_name: "embedding/'+model_name+'"\n        metadata_path: "embedding_labels.tsv"\n    }'
    handler.write(config)
    
print("embedding_initialization_finish")


# In[36]:


saver = tf.compat.v1.train.Saver(max_to_keep=3)
summ_op = tf.compat.v1.summary.merge(summ_training)


# In[37]:


print("\n----------------------- main function ----------------------------------")
# Config to turn on JIT compilation
config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter(log_dir, graph=sess.graph)
    writer.add_graph(sess.graph)
    
    run_round = 2000000  
    print("----------------------- start ----------------------------------")      
    for i in range(run_round):
        print("%d / %d" %(i,run_round), end="\r")
        batch_index = np.random.choice(word_training_tags.shape[0], batch_size, replace=False)
        _ = sess.run(train_op, feed_dict={input_image: word_training_features[batch_index], 
                                          input_label: word_training_tags[batch_index]})
        if i % 500 == 0:
            s = sess.run(summ_op, feed_dict={input_image: word_training_features[batch_index], 
                                             input_label: word_training_tags[batch_index]})
            writer.add_summary(s, i)
            # for testing
            batch_index = np.random.choice(word_testing_tags.shape[0], batch_size, replace=False)
            [predict_, correct_prediction_, test_acc_, testing_op_] = sess.run([predict, correct_prediction, test_acc, testing_op],
                                                                                feed_dict={input_image: word_training_features[batch_index],
                                                                                           input_label: word_training_tags[batch_index]})
        
            print("------------ rounds: %d --------------" %(i))
            print("acc: ",test_acc_)
            print("predict: ",predict_)
            print(correct_prediction_)          
            writer.add_summary(testing_op_, i)

        if i % 10000 == 0:
            sess.run(assignment, feed_dict={input_image: word_training_features[:embedding_batch], 
                                            input_label: word_training_tags[:embedding_batch]})
            saver.save(sess, log_dir + model_name+".ckpt", i)
            
print("-----------------------------------------------------------------------")

print("training finish")


# In[ ]:




