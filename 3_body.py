import  requests
#参数请求
#定义url
# base_url = 'http://192.168.1.101:8000'
# base_url = 'http://192.168.43.14:8000'
base_url = "http://127.0.0.1:5000"
# post方法
# 定义body中传递的数据
from_date = {'picPath':'/Users/lixizi/Documents/project/project_file/server/oblique_wrist_arm_51.jpg'
             ,'password':'123456'}
p = requests.post(base_url+'/api', data=from_date)

# 打印返回的数据
print(p.text)