#!/usr/bin/python3

def method2(md):
    print("test")
    print(md)



sites = ["Baidu", "Google", "Runoob", "Taobao"]
for site in sites:
    if site == "Runoob":
        print("菜鸟教程!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")

md = 17.5
method2(md)
print("结束")
