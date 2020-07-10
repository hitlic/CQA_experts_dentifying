#author: "xian"
#date: 2018/5/30
import re
import requests

#使用requests库的会话维持用法
def start_get_session():
    s = requests.session()
    return s

def get_base_cookies(s):
    s.get('https://user.qunar.com/passport/login.jsp')
    get_image(s)
    s.get('https://user.qunar.com/passport/addICK.jsp?ssl')
    response = s.get('https://rmcsdf.qunar.com/js/df.js?org_id=ucenter.login&js_type=0')

    #获取sessionid
    session_id = re.findall(r'sessionId=(.*?)&',response.text)
    session_id = session_id[0]  #脱壳操作

    #获取fid
    s.get('https://rmcsdf.qunar.com/api/device/challenge.json?callback=callback_1527735086394&sessionId={}&domain=qunar.com&orgId=ucenter.login'.format(session_id)) 
    s.cookies.update({'QN271':session_id})



#获取图片 
def get_image(s): 
    response = s.get('https://user.qunar.com/captcha/api/image?k={en7mni(z&p=ucenter_login&c=ef7d278eca6d25aa6aec7272d57f0a9a&t=1527644979725')
    with open('./img/code.png','wb') as f: 
        f.write(response.content) 


#登录函数 
def login(s,username,password,code):
    data = {
    'loginType': 0,
    'username': username,
    'password': password,
    'remember': 1,
    'vcode': code,
    }
    url = 'https://user.qunar.com/passport/loginx.jsp'
    response = s.post(url,data = data)
    print(response.text)
    response = s.get('http://user.qunar.com/index/basic') 
#模拟登录后爬取该网页
    print(response.text) 




#主函数 
if __name__ == '__main__':
    session = start_get_session()
    get_base_cookies(session)
    username = input('请输入用户名：')
    password = input('请输入密码：')
    code = input('请输入验证码：')
    login(session,username,password,code)