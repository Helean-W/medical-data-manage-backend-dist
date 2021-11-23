1. 新建python3虚拟环境并激活
2. 安装mysql apt-get update apt-get install mysql-server apt-get install mysql-client
3. 安装python的mysql驱动 apt-get install python3-dev libmysqlclient-dev
4. import MySQLdb测试是否安装成功
5. 根据setting创建数据库，附带default charset utf8
6. 安装所需套件 pip install -r 'requirements.txt'
7. 迁移数据表 python manage.py makemigrations python manage.py migrate

8. 新建目录结构 
    resources
        dcmTemp  diagnosis  grading_diag  jpgTemp  zipTemp
                    mask  temp
                            mask

9. 新建文件: uwsgi.log  uwsgi.pid
10. uwsgi部署 uwsgi --ini uwsgi.ini