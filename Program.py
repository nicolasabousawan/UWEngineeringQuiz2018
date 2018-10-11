
# coding: utf-8

# In[3]:


# Prof Lukasz Golab
# Nicolas Abou Sawan
# Data Science Lab
# Naive Bayes decision model for Engineering Quiz


import numpy as np

#liklihood tables
classPriorProbability = np.array([0.064164649,0.06779661,0.059322034,0.047215496,0.03874092,0.024213075,0.00968523,
                                  0.119854722,0.099273608,0.182808717,0.071428571,0.161016949,0.054479419], dtype="d")

                                  
fieldInterestArray = np.array([  
    [0.017241379,0.032786885,0.074074074,0.022727273,0.189189189,0.04,0.230769231,0.240384615,0.022988506,0.025641026,0.03125,0.007246377,0.08], #Finance
    [0.155172414,0.639344262,0.074074074,0.045454545,0.162162162,0.08,0.076923077,0.298076923,0.494252874,0.269230769,0.1875,0.007246377,0.14], #Manufacturing  
    [0.172413793,0.016393443,0.055555556,0.772727273,0.324324324,0.04,0.076923077,0.346153846,0.068965517,0.455128205,0.046875,0.942028986,0.62], #Software Development
    [0.620689655,0.262295082,0.074074074,0.136363636,0.243243243,0.44,0.230769231,0.076923077,0.298850575,0.224358974,0.71875,0.036231884,0.1], # Research
    [0.034482759,0.049180328,0.722222222,0.022727273,0.081081081,0.4,0.384615385,0.038461538,0.114942529,0.025641026,0.015625,0.007246377,0.06] #Construction
], dtype="d" )

teamResponsibleArray = np.array([
    [0.06779661,0.080645161,0.054545455,0.088888889,0.078947368,0.038461538,0.142857143,0.19047619,0.477272727,0.159235669,0.046153846,0.035971223,0.039215686], #The Automobile
    [0.06779661,0.064516129,0.054545455,0.6,0.289473684,0.115384615,0.071428571,0.466666667,0.079545455,0.108280255,0.107692308,0.726618705,0.392156863], # The internet
    [0.016949153,0.048387097,0.763636364,0.022222222,0.052631579,0.153846154,0.571428571,0.066666667,0.068181818,0.01910828,0.030769231,0.007194245,0.117647059], # CN tower
    [0.084745763,0.338709677,0.036363636,0.066666667,0.315789474,0.5,0.071428571,0.142857143,0.204545455,0.070063694,0.384615385,0.035971223,0.215686275], #Solar Panels
    [0.220338983,0.016129032,0.036363636,0.2,0.236842105,0.038461538,0.071428571,0.038095238,0.147727273,0.617834395,0.076923077,0.165467626,0.176470588], # the robot
    [0.542372881,0.451612903,0.054545455,0.022222222,0.026315789,0.153846154,0.071428571,0.095238095,0.022727273,0.025477707,0.353846154,0.028776978,0.058823529] #penicillin
], dtype="d")

startWorkArray = np.array([
    [0.267857143,0.542372881,0.326923077,0.404761905,0.314285714,0.173913043,0.454545455,0.490196078,0.423529412,0.311688312,0.35483871,0.404411765,0.208333333], #large corp
    [0.482142857,0.338983051,0.576923077,0.261904762,0.571428571,0.739130435,0.454545455,0.333333333,0.435294118,0.38961039,0.387096774,0.367647059,0.5], # small comp
    [0.25,0.118644068,0.096153846,0.333333333,0.114285714,0.086956522,0.090909091,0.176470588,0.141176471,0.298701299,0.258064516,0.227941176,0.291666667], # start up
], dtype="d")

designSomethingArray = np.array([
    [0.452830189,0.410714286,0.204081633,0.666666667,0.3125,0.4,0.25,0.797979798,0.12195122,0.245033113,0.813559322,0.804511278,0.577777778], # non tangible
    [0.547169811,0.589285714,0.795918367,0.333333333,0.6875,0.6,0.75,0.202020202,0.87804878,0.754966887,0.186440678,0.195488722,0.422222222], # tangible
], dtype="d")

workPartOfArray = np.array([
    [0.339622642,0.196428571,0.265306122,0.41025641,0.46875,0.25,0.375,0.191919192,0.292682927,0.298013245,0.254237288,0.526315789,0.044444444], #focused
    [0.660377358,0.803571429,0.734693878,0.58974359,0.53125,0.75,0.625,0.808080808,0.707317073,0.701986755,0.745762712,0.473684211,0.955555556], #Multi
], dtype="d")

onProjectArray = np.array([
    [0.528301887,0.75,0.816326531,0.564102564,0.6875,0.6,0.875,0.484848485,0.646341463,0.642384106,0.593220339,0.526315789,0.422222222], #know it all
    [0.471698113,0.25,0.183673469,0.435897436,0.3125,0.4,0.125,0.515151515,0.353658537,0.357615894,0.406779661,0.473684211,0.577777778], #continuous 
], dtype="d")

careMostArray = np.array([
    [0.924528302,0.410714286,0.326530612,0.128205128,0.21875,0.05,0.125,0.282828283,0.256097561,0.271523179,0.474576271,0.278195489,0.311111111,0.328087167], #Health
    [0.01754386,0.45,0.377358491,0.162790698,0.305555556,0.708333333,0.583333333,0.145631068,0.430232558,0.290322581,0.301587302,0.160583942,0.265306122], #Env
    [0.070175439,0.066666667,0.188679245,0.186046512,0.194444444,0.125,0.166666667,0.310679612,0.127906977,0.083870968,0.079365079,0.167883212,0.224489796], #Human Rights
    [0.035087719,0.083333333,0.113207547,0.511627907,0.277777778,0.083333333,0.083333333,0.262135922,0.186046512,0.35483871,0.158730159,0.394160584,0.204081633], #tech
])

eavesdropArray = np.array([
    [0.210526316,0.416666667,0.20754717,0.069767442,0.277777778,0.125,0.166666667,0.155339806,0.639534884,0.24516129,0.333333333,0.131386861,0.142857143], # metals
    [0.192982456,0.216666667,0.603773585,0.139534884,0.083333333,0.666666667,0.666666667,0.233009709,0.093023256,0.025806452,0.111111111,0.072992701,0.306122449], #earthquakes
    [0.385964912,0.116666667,0.094339623,0.488372093,0.305555556,0.166666667,0.083333333,0.223300971,0.058139535,0.219354839,0.444444444,0.386861314,0.224489796], # microchip
    [0.210526316,0.25,0.094339623,0.302325581,0.333333333,0.041666667,0.083333333,0.388349515,0.209302326,0.509677419,0.111111111,0.408759124,0.326530612], # lean production 
], dtype="d")

workingWithArray = np.array([
    [0.666666667,0.666666667,0.773584906,0.23255814,0.472222222,0.666666667,0.583333333,0.572815534,0.418604651,0.277419355,0.587301587,0.226277372,0.734693878], #People
    [0.192982456,0.066666667,0.132075472,0.697674419,0.361111111,0.166666667,0.166666667,0.106796117,0.151162791,0.470967742,0.206349206,0.759124088,0.163265306], #Comp
    [0.052631579,0.066666667,0.018867925,0.023255814,0.055555556,0.041666667,0.166666667,0.300970874,0.023255814,0.012903226,0.047619048,0.00729927,0.081632653], #Business    
    [0.087719298,0.2,0.075471698,0.046511628,0.111111111,0.125,0.083333333,0.019417476,0.406976744,0.238709677,0.158730159,0.00729927,0.020408163], #Machine
], dtype="d")

#engineering programs in alpha order
programs = [ "Biomedical Engineering", "Chemical Engineering" , "Civil Engineering", "Computer Engineering" , "Electrical Engineering" , "Environmental Engineering", "Geological Engineering", "Management Engineering" , "Mechanical Engineering" , "Mechatronics Engineering", "Nanotechnology Engineering", "Software Engineering" , "Systems Design Engineering" ]


# Variables take in responses of the values from the quiz
# For example if there are 4 options [possible values assigned: 0,1,2,3]
# If the user pick the first option of the four
# the variable should be assigned the integer of 0

def Outcomes(field_interest, team_responsible, start_work, design_something, 
             work_partof, on_project, care_most, eavesdrop, working_with):
    probabilities = []
    for index in range(0,13):
        #computing the naive bayes proability for each program 
        outcome = (classPriorProbability[index] * fieldInterestArray[field_interest][index] *
                    teamResponsibleArray[team_responsible][index] * startWorkArray[start_work][index] *
                    designSomethingArray[design_something][index] * workPartOfArray[work_partof][index] *
                    onProjectArray[on_project][index] * careMostArray[care_most][index] * eavesdropArray[eavesdrop][index] *
                    workingWithArray[working_with][index])
        probabilities.append(outcome) # add probability to a list so all values can be compared afterwards
    #finding the largest 2 values 
    testPoint = 0 # value tester
    answer=0 #index counter 
    for values in range(13):
        if testPoint < probabilities[values]:
            answer = values
            testPoint = probabilities[values]
    probabilities[answer]=0 #largest value becomes 0 so the second largest value is fount
    testPoint=0
    answer1=0
    for values in range(12):
        if testPoint  < probabilities[values]:
            answer1 = values
            testPoint = probabilities[values]
    print (programs[answer], programs[answer1])


# In[5]:




