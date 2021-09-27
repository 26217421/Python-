import os
import time
# 1. 需要备份的文件与目录将被
# 指定在一个列表中。
# 例如在 Windows 下：
source = ['"D:\\My Documents"']
# 在这里要注意到我们必须在字符串中使用双引号
# 用以括起其中包含空格的名称。
# 在 Mac OS X 与 Linux 下：
# source = ['/Users/swa/notes']

target_dir = 'E:\\Backup'
# 在 Mac OS X 和 Linux 下：
# target_dir = '/Users/swa/backup'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

today = target_dir + os.sep + time.strftime('%Y%m%d')
# 将当前时间作为 zip 文件的文件名
now = time.strftime('%H%M%S')

comment = input('Enter a comment --> ')
# 检查是否有评论键入
if len(comment) == 0:
    target = today + os.sep + now + '.zip'
else:
    target = today + os.sep + now + '_' + \
        comment.replace(' ', '_') + '.zip'
# 如果子目录尚不存在则创建一个
if not os.path.exists(today):
    os.mkdir(today)
print('Successfully created directory', today)

# 5. 我们使用 zip 命令将文件打包成 zip 格式
zip_command = 'zip -r {0} {1}'.format(target, ' '.join(source))
# 运行备份
print('Zip command is:')
print(zip_command)
print('Running:')
if os.system(zip_command) == 0:
    print('Successful backup to', target)
else:
    print('Backup FAILED')
