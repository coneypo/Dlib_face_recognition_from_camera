import pymysql

db = pymysql.connect("localhost", "root", "intel@123", "dlib_database")
cursor = db.cursor()

# 1. get database face numbers
cmd_rd = "select count(*) from dlib_face_table;"
cursor.execute(cmd_rd)
results = cursor.fetchall()
person_cnt = int(results[0][0])

# 2. get features for person X
for person in range(person_cnt):
    # lookup for personX
    cmd_lookup = "select * from dlib_face_table where person_x=\"person_"+str(person+1)+"\";"
    cursor.execute(cmd_lookup)
    results = cursor.fetchall()
    results = list(results[0][1:])
    print(results)