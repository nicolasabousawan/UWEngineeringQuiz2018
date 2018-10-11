
# coding: utf-8

# In[8]:


import numpy as np
from NaiveBayesModel import MB

data=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape(1,-1)

field_interest = input("What field interests you the most?" "\n 1. Finance" "\n 2. Manufacturing" 
                       "\n 3. Software Development" "\n 4. Research" " \n 5. Construction \n")
team_responsible = input("I would like to have been part of the team that was responsible for:\n" "1. The Automobile \n 2. The Internet \n 3. The CN Tower \n 4. Solar Panels \n 5. The Robot \n 6. Penicillin \n")
start_work= input( "When you start work, ideally you would want to... \n 1. Work for a large corporation \n 2. Work for a small company \n 3. Work in a start up \n")
design_something = input('I would rather design something that... \n 1. I can see it function and prove it works, but can\'t necessary touch \n 2. I can see function and tangibly touch the parts\n')
work_partof = input("I would rather work as part of a... \n 1. Focused team \n 2. Multidisciplinary team \n")
on_project = input("When working on a project would you rather \n 1. Know all the requirements for the project at the beginning \n 2. Continuously change and update requirements as you go \n")
care_most = input("What cause do you care about the most? \n 1. Health \n 2. Environment  \n 3. Human Rights \n 4. Bringing technology to developing countries \n")
eavesdrop = input("""You are most likely to eavesdrop in a conversation regarding \n 1. A groundbreaking metal that will create stronger cars, planes and spaceships for a cheaper cost 
2. A new way to deal with movement in the Earth’s surface to protect buildings from earthquakes 
3. A microchip that is smaller than usual but twice as powerful as before 
4. A factory that is able to 100% rely on robotics while producing no defected products\n""")
working_with = input("""In an ideal setting, you are working with: \n 1. People \n 2. Computers \n 3. Business Processes \n 4. Machinery \n""")

data[0][int(field_interest)-1] = 1
data[0][5]=1
data[0][int(team_responsible)-1 + 5] = 1
data[0][int(start_work)-1+5+6] = 1
data[0][int(design_something)-1+5+6+3] = 1
data[0][int(work_partof)-1+5+6+3+2] = 1
data[0][int(on_project)-1+5+6+3+2+2] = 1
data[0][int(care_most)-1+5+6+3+2+2+2] = 1
data[0][int(eavesdrop)-1+5+6+3+2+2+2+4] = 1
data[0][int(working_with)-1+5+6+3+2+2+2+4+4] = 1

#print(MB.predict(data))

print(MB.predict(data))
probabilities = MB.predict_proba(data)
programs = ["Biomedical Engineering", "Chemical Engineering", "Civil Engineering", "Computer Engineering" ," Electrical Engineering", "Environmental Engineering", "Geological Engineering", "Management Engineering", "Mechanical Engineering", "Mechatronics Engineering", "Nanotechnology Engineering" ,"Software Engineering", "Systems Design Engineering"]

def Max(array):
    max = 0
    answer=0
    for values in range(12):
        if max < array[0][values]:
            max = array[0][values]
            answer = values
    array[0][answer] = 0
########################################
    max =0
    for values in range(12):
        if max < array[0][values]:
            max= array[0][values]
            answer1 = values
    array[0][answer1] = 0
    return (answer, answer1)

answer, answer1 = Max(probabilities)

print(programs[answer], programs[answer1])

