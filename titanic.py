# 开发者：袁策
# 开发日期：2023/10/13
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#处理dataframe基本方法
file=pd.read_csv('train.csv')       #读取表格
print(file)     #打印原表格
n=len(file[file.columns[0]])        #统计总行数/人数
print(n)
print(file[0:4])        #取表格的前四行
print(file.Survived)        #打印存活列

#性别模块
women=file[file.Sex=="female"]["Survived"]      #取所有女性的存活情况,结果是series
men=file[file.Sex=="male"]["Survived"]          #取所有男性的存活情况,结果是series
'''
women=file.loc[file.Sex=="female"]["Survived"]      #取所有女性的存活情况,结果是series，与上方语句相同用处
'''
women_survived=women[women.values==1]       #取所有存活的女性
men_survived=men[men.values==1]       #取所有存活的男性
#以女性为例
print(women)
print(women_survived)
print(women_survived[women_survived.index==3]) #取表格中顺序第三个存活的女性的编号
women_survived_rate=women_survived.sum()/len(women)         #女性存活率
men_survived_rate=men_survived.sum()/len(men)       #男性存活率
print(women_survived_rate)
print(men_survived_rate)
print(pd.DataFrame({"survived":[men_survived_rate,women_survived_rate],"died":[1-men_survived_rate,1-women_survived_rate]},index=["men","women"]))     #创建一个dataframe表示性别及其存活情况

#性别存活绘图模块
#图1
x_label=["men_total","men_survived","women_total","women_survived"]     #设置x轴
y_label=[len(men),len(men_survived),len(women),len(women_survived)]     #设置y轴
color=["red","blue","red","blue"]       #设置颜色
plt.bar(x_label,y_label,width=1,color=color)        #基本参量
plt.legend()        #不显示图例
plt.show()          #显示绘图

#得出结果与现实匹配度低，男性在灾难中应该存活率较高，单凭性别无法判断，需要加深决策树的深度，因此再添加几个特征

#根据猜想选取几个可能有用的特征
features=["Pclass","Sex","Age","SibSp","Parch","Survived"]         #选取的特征
features_show=file[features]        #从表格中提取特征栏
print(features_show)
print(features_show.describe())         #获取特征的基本数据
#可以看到Age列缺少了部分数据，方案有2种,1：用平均年龄进行填充 2：删除  我选择用方案1
age_mean=features_show["Age"].mean()
features_show["Age"]=features_show["Age"].fillna(age_mean)
print(features_show.describe())         #缺失值处理完毕
#以此类推得出其他特征的图像

#座位等级图像
Pclass1=features_show[features_show["Pclass"]==1]       #仓位1的人
Pclass1_survived=Pclass1[Pclass1["Survived"]==1]        #仓位1活着的人
print(Pclass1_survived)
Pclass1_survived_rate=len(Pclass1_survived)/len(Pclass1)      #仓位1存活率
print(Pclass1_survived_rate)
#以此类推
Pclass2=features_show[features_show["Pclass"]==2]
Pclass2_survived=Pclass2[Pclass2["Survived"]==1]
print(Pclass2_survived)
Pclass2_survived_rate=len(Pclass2_survived)/len(Pclass2)
print(Pclass2_survived_rate)
Pclass3=features_show[features_show["Pclass"]==3]
Pclass3_survived=Pclass3[Pclass3["Survived"]==1]
print(Pclass3_survived)
Pclass3_survived_rate=len(Pclass3_survived)/len(Pclass3)
print(Pclass3_survived_rate)

#座位等级绘图模块
#图2
x_label=["Pyclass1_total","Pyclass1_survived","Pyclass2_total","Pyclass2_survived","Pyclass3_total","Pyclass3_survived"]
y_label=[len(Pclass1),len(Pclass1_survived),len(Pclass2),len(Pclass2_survived),len(Pclass3),len(Pclass3_survived)]
color=["red","blue"]*3
plt.bar(x_label,y_label,width=0.5,color=color)
plt.legend()
plt.show()
#可以看到此时绘制柱状图会导致行标签发生重叠，来改进下柱形图，便于比较

#绘制座位等级柱形图plus
x_label=["Pclass1","Pclass2","Pclass3"]         #行标签
total=[len(Pclass1),len(Pclass2),len(Pclass3)]          #数据-总人数
survived=[len(Pclass1_survived),len(Pclass2_survived),len(Pclass3_survived)]        #数据-存活人数
length=np.arange(len(x_label))          #使用numpy创建一个列表[1,2,3]
width=0.35      #设置柱形的宽度
fig,ax=plt.subplots()       #图为一行一列（一张图），此为第一张
ax.bar(length-width/2,total,width,label="total")        #使人数柱位于x轴标签的左边，图例为"total"
ax.bar(length+width/2,survived,width,label="survived")      #使存活人数柱位于x轴标签的右边，图例为"survived"
ax.set_xlabel("Pclass level")
ax.set_ylabel('number')         #设置列标签为"number"
ax.set_title('total number and survived number')        #设置图标名称
ax.set_xticks(length)       #设置x轴标签的数量
ax.set_xticklabels(x_label)         #设置x轴标签的名称
ax.legend()        #显示图例
plt.show()      #显示图表
#由图表得，座位等级越高，生存几率越大，符合现实及预测

#年龄模块，用直方图来绘制
#图3
age_survived=features_show['Age'][features_show['Survived']==1]         #获取存活人的年龄，类型是series
age_died=features_show['Age'][features_show['Survived']==0]         #获取死亡人的年龄
plt.hist([age_survived,age_died],stacked=True,label=['survived','died'],bins=8)        #表格基本数据(stacked=True表示可以叠加数据),将年龄分成8份
plt.legend()
plt.title('Age_Survived')
plt.show()
print(age_survived)
#这样拟合年龄不便于将其与其他变量进行合并处理，在此将年龄变量进行转换

#年龄转换模块
def age_change(age):
    if age<=18:         #青年编号为1
        return 1
    if 18<age<50:       #中年编号为2
        return 2
    if 50<=age:         #老年编号为3
        return 3
for i in range(len(age_survived)):      #遍历series
    age_survived.iloc[i]=age_change(age_survived.iloc[i])      #替换值
for i in range(len(age_died)):
    age_died.iloc[i]=age_change(age_died.iloc[i])
#绘图模块与上面一致
#图4
plt.hist([age_survived,age_died],stacked=True,label=['survived','died'],bins=3)
plt.title('Age_Survived')
plt.show()
print(age_survived)

#结合性别和年龄预测哪种组合生存率最高
for i in range(n):
    features_show.iloc[i,2]=age_change(features_show.iloc[i,2])
def sex_change(sex):        #性别编码函数
    if sex=="female":
        return(0)
    if sex=="male":
        return(1)
for i in range(n):
    features_show.iloc[i,1]=sex_change(features_show.iloc[i,1])
#女性组
team1=features_show[features_show["Sex"]==0]        #语句结构一样，看一组就行
team1=team1[team1["Age"]==1.0]
team1_survived=team1[team1["Survived"]==1]
team1_survived_rate=len(team1_survived)*100/len(team1[team1.columns[0]])

team2=features_show[features_show["Sex"]==0]
team2=team2[team2["Age"]==2.0]
team2_survived=team2[team2["Survived"]==1]
team2_survived_rate=len(team2_survived)*100/len(team2[team2.columns[0]])

team3=features_show[features_show["Sex"]==0]
team3=team3[team3["Age"]==3.0]
team3_survived=team3[team3["Survived"]==1]
team3_survived_rate=len(team3_survived)*100/len(team3[team3.columns[0]])
#男性组
team4=features_show[features_show["Sex"]==1]
team4=team4[team4["Age"]==1.0]
team4_survived=team4[team4["Survived"]==1]
team4_survived_rate=len(team4_survived)*100/len(team4[team4.columns[0]])

team5=features_show[features_show["Sex"]==1]
team5=team5[team5["Age"]==2.0]
team5_survived=team5[team5["Survived"]==1]
team5_survived_rate=len(team5_survived)*100/len(team5[team5.columns[0]])

team6=features_show[features_show["Sex"]==1]
team6=team6[team6["Age"]==3.0]
team6_survived=team6[team6["Survived"]==1]
team6_survived_rate=len(team6_survived)*100/len(team6[team6.columns[0]])

#图5
x_label1=["(0,1)","(0,2)","(0,3)","(1,1)","(1,2)","(1,3)"]   #设置x轴
y_label1=[team1_survived_rate,team2_survived_rate,team3_survived_rate,team4_survived_rate,team5_survived_rate,team6_survived_rate]    #设置y轴
for a,b in zip(x_label1,y_label1):   #柱子上的数字显示
    plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
plt.bar(x_label1,y_label1,width=0.5)        #基本参量
plt.title("Age-Sex survived rate")
plt.legend()        #不显示图例
plt.show()          #显示绘图
#性别对生存率影响比年龄大




