
org_train_file = 'dataset/training.1600000.processed.noemoticon.csv'
org_test_file = 'dataset/testdata.manual.2009.06.14.csv'
# import io
# import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')

# 提取文件中有用的字段
def usefull_filed(org_file, output_file):
    output = open(output_file, 'w')
    with open(org_file, buffering=10000,encoding='latin-1') as f:
        for line in f:  # "4","2193601966","Tue Jun 16 08:40:49 PDT 2009","NO_QUERY","AmandaMarie1028","Just woke up. Having no school is the best feeling ever "
            line = line.replace('"', '')
            clf = line.split(',')[0]  # 4
            if clf == '0':
                clf = [0, 0, 1]  # 消极评论
            elif clf == '4':
                clf = [1, 0, 0]  # 积极评论

            tweet = line.split(',')[-1]
            outputline = str(clf) + ':%:%:%:' + tweet
            try:

                output.write(outputline)  # [0, 0, 1]:%:%:%: that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D
            except:
                continue
    output.close()  # 处理完成，处理后文件大小127.5M


usefull_filed(org_train_file, 'dataset/training.csv')
usefull_filed(org_test_file, 'dataset/tesing.csv')