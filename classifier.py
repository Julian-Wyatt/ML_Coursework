# In order to run this classifier, ensure you have the following packages installed
# (although this code will attempt to install those missing *Checked on a UNIX based system)
# package checklist
# pandas, numpy, sklearn, seaborn, matplotlib, hyperopt
# Once these packages are installed - execute the code in terminal with python classifier.py
# or through an IDE
# Also note, the code will check for the data CVS's within the Dataset folder, if this fails, it will check in the cwd
# Similarly for saved figures, these will be saved in ./Saved_Figs - if this fails they will be saved in the cwd

# random state seed to ensure repeatable metrics
random_state = 1024

# runs a command line process to install a package if an ImportError occurs
import subprocess
def install(package):
    subprocess.run([sys.executable, "-m", "pip", "install", package, "--user"])


try:
    import pandas
except ImportError:
    install("pandas")
    import pandas

try:
    from sklearn import *
    from sklearn import model_selection
    from sklearn import utils
except ImportError:
    install("sklearn")
    from sklearn import *

from sklearn import model_selection
from sklearn import utils

# import default python modules
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')   # ignore division by 0 warning when measuring precision in RF hyper tuning

# import numpy & matplotlib
try:
    import numpy as np
except ImportError:
    install('numpy')
    import numpy as np
try:
    import matplotlib as mpl
except ImportError:
    install('matplotlib')
    import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt

# import seaborn for heatmap
try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

# import hyperopt for hyperparameter optimisation using bayesian optimisation
try:
    import hyperopt
except ImportError:
    install("hyperopt")
    import hyperopt

# general functions for saving figures and plotting graphs with matplotlib
def saveFigure(name):
    """
    Saves figure to the Save_Figs folder
    :param name: name of the file
    :return: A saved figure of the scatter to the saved_figs folder
    """

    # stores the figure in the saved figures folder for ease of access and clutter
    if "Saved_Figs" in os.listdir(os.getcwd()):
        plt.savefig(os.path.join(path, "Saved_Figs", (name.replace(" ","_")) + ".png"), format="png", dpi=400, bbox_inches='tight')
    # however, if the file does not exist - it outputs to cwd
    else:
        plt.savefig(os.path.join(path, (name.replace(" ","_")) + ".png"), format="png", dpi=400, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()


def plotSave(name, dataX, nameX, dataY, nameY):
    """
    Plots the data to a matplotlib scatter
    :param name: name of the graph
    :param dataX: list of x data points
    :param nameX: name of the x data points
    :param dataY: list of y data points
    :param nameY: name of the y data points
    :return: A saved figure of the scatter to the saved_figs folder
    """

    plt.scatter(dataX, dataY, color='r', alpha=0.1)
    plt.xlabel(nameX,fontsize=10)
    plt.ylabel(nameY,fontsize=10)
    # plt.title(name,fontsize=17)
    saveFigure(name)

# sets pandas output to display all columns
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

#####################---DATA GATHERING---#####################

# start by getting the time to check time taken
start = time.time()

# set current working directory
path = os.getcwd() + "/"

def dataGather():
    '''
    Gather data from the csv's stored in ./Dataset/ or in ./
    :return: modelData - ie the gathered information stored about each row in stuInfo
    '''
    # try looking for the data csv files in the Dataset folder
    try:
        stuVLE = pandas.read_csv(path + "Dataset/studentVle.csv", na_values=['no info', ' '], header=0)
        stuInfo = pandas.read_csv(path + "Dataset/studentInfo.csv", na_values=['no info', ' '], header=0)
        stuAssessments = pandas.read_csv(path + "Dataset/studentAssessment.csv", na_values=['no info', ' '], header=0)
        assessments = pandas.read_csv(path + "Dataset/assessments.csv", na_values=['no info', ' '], header=0)
        courses = pandas.read_csv(path + "Dataset/courses.csv", na_values=['no info', ' '], header=0)
    # however this folder may not exist during submission so look in cwd instead
    except FileNotFoundError:
        stuVLE = pandas.read_csv(path + "studentVle.csv", na_values=['no info', ' '], header=0)
        stuInfo = pandas.read_csv(path + "studentInfo.csv", na_values=['no info', ' '], header=0)
        stuAssessments = pandas.read_csv(path + "studentAssessment.csv", na_values=['no info', ' '], header=0)
        assessments = pandas.read_csv(path + "assessments.csv", na_values=['no info', ' '], header=0)
        courses = pandas.read_csv(path + "courses.csv", na_values=['no info', ' '], header=0)

    # firstly look towards the vle stats
    # so I collected the total number of times they clicked on their portal

    tempVLE = stuVLE[["id_student", "sum_click"]]

    # then summed all of that data across the time they spent studying

    visits = tempVLE.groupby("id_student").sum()
    # sets id_student as a column rather than primary key for table
    visits.reset_index(level=0, inplace=True)

    # sum click stats
    # get the variance of the sum_click of a given module
    assessment_var = stuVLE.groupby(["code_module", "code_presentation"])["sum_click"].var() \
        .reset_index(level=1).rename(columns={"sum_click": "sum_click_assessment_var"})

    stuInfo = stuInfo.merge(assessment_var, how="outer", on=["code_module", "code_presentation"])

    # get the mean of the sum_click of a given module
    assessment_mean = stuVLE.groupby(["code_module", "code_presentation"])["sum_click"].mean() \
        .reset_index(level=1).rename(columns={"sum_click": "sum_click_assessment_mean"})
    stuInfo = stuInfo.merge(assessment_mean, how="outer", on=["code_module", "code_presentation"])

    # get the mad of the sum_click of a given module
    assessment_mad = stuVLE.groupby(["code_module", "code_presentation"])["sum_click"].mad() \
        .reset_index(level=1).rename(columns={"sum_click": "sum_click_assessment_mad"})
    stuInfo = stuInfo.merge(assessment_mad, how="outer", on=["code_module", "code_presentation"])

    # get the med of the sum_click of a given module
    assessment_med = stuVLE.groupby(["code_module", "code_presentation"])["sum_click"].median() \
        .reset_index(level=1).rename(columns={"sum_click": "sum_click_assessment_med"})
    stuInfo = stuInfo.merge(assessment_med, how="outer", on=["code_module", "code_presentation"])

    # get the std of the sum_click of a given module
    assessment_std = stuVLE.groupby(["code_module", "code_presentation"])["sum_click"].std() \
        .reset_index(level=1).rename(columns={"sum_click": "sum_click_assessment_std"})
    stuInfo = stuInfo.merge(assessment_std, how="outer", on=["code_module", "code_presentation"])

    # get the variance of the sum_click of a given students portal visits
    stuVar = stuVLE.groupby("id_student")["sum_click"].var() \
        .reset_index(level=0).rename(columns={"sum_click": "sum_click_student_var"})
    stuInfo = stuInfo.merge(stuVar, how="outer", on=["id_student"])

    # get the std of the sum_click of a given students portal visits
    stuStd = stuVLE.groupby("id_student")["sum_click"].std() \
        .reset_index(level=0).rename(columns={"sum_click": "sum_click_student_std"})
    stuInfo = stuInfo.merge(stuStd, how="outer", on=["id_student"])

    # get the mad of the sum_click of a given students portal visits
    stuMad = stuVLE.groupby("id_student")["sum_click"].mad() \
        .reset_index(level=0).rename(columns={"sum_click": "sum_click_student_mad"})
    stuInfo = stuInfo.merge(stuMad, how="outer", on=["id_student"])

    # get the median of the sum_click of a given students portal visits
    stuMed = stuVLE.groupby("id_student")["sum_click"].median() \
        .reset_index(level=0).rename(columns={"sum_click": "sum_click_student_med"})
    stuInfo = stuInfo.merge(stuMed, how="outer", on=["id_student"])

    # function to convert a ranged value to the median value of that range
    # based on the regex - ([0-9]*)\-([0-9]*)(%?)
    def rangeToMed(m):
        """
        function to convert a ranged value to the median value of that range
        based on the regex - ([0-9]*)\-([0-9]*)(%?)
        :param m: the value of a row in the table
        :return: the altered median value
        """
        try:
            return str((int(m.group(1)) + int(m.group(2))) / 2)
        # return (int(m.group(1)) + int(m.group(2))) / 2
        except ValueError:
            print("error occured on " + str(m.group(0)))


    # uses the function above to swap the range to the median as a numerical value,
    # while filling empty values as the middle range
    stuInfo["imd_band"] = stuInfo["imd_band"].str.replace(r'([0-9]*)\-([0-9]*)(%?)', rangeToMed).astype(
        "float32").fillna(50)

    # swaps the age band ranges to rough estimates of the mean
    stuInfo["age_band"] = stuInfo["age_band"].replace({"0-35": 25, "35-55": 45, "55<=": 60})

    # adds the module length to the student info
    stuInfo = stuInfo.merge(courses, on=["code_module", "code_presentation"])

    # make assessment_types numeric for analysis
    assessments["assessment_type"] = assessments["assessment_type"].astype("category").cat.codes

    # performs an inner join on the two tables to get all of the information regarding the assessment
    # as well as the individual student scores
    all_Assessments = stuAssessments.merge(assessments)

    # if someone has banked their data, set their date submitted as the median to prevent skew, as it originally used 0
    temp = all_Assessments.where(all_Assessments["is_banked"] == 1)

    # find those medians
    temp1 = pandas.DataFrame(all_Assessments.groupby("id_assessment")["date_submitted"].median())
    temp1.reset_index(level=0, inplace=True)
    # updates the date submitted for the isbanked assessments
    temp = temp.update(temp1)
    # update this for all assessments
    all_Assessments.update(temp)

    # date is numerical value signifying how many days after the start of the course the coursework is due
    # date submitted is the days after the course started
    # therefore total days early is sum of date - sum of date submitted

    # get the total due dates
    submissionDates = all_Assessments.groupby("id_student")["date"].sum()
    # get the sum of the dates submitted
    days_submiited = all_Assessments.groupby("id_student")["date_submitted"].sum()
    # make substitution
    daysEarly = submissionDates.sub(days_submiited)
    daysEarly = daysEarly.rename("daysEarly").to_frame()
    daysEarly.reset_index(level=0, inplace=True)

    stuInfo = stuInfo.merge(daysEarly, how="outer", on=["id_student"])

    # total number of banked assessments for a given student
    stuInfo = stuInfo.merge(
        pandas.DataFrame(all_Assessments.groupby("id_student")["is_banked"].sum()).reset_index(level=0), how="outer", on=["id_student"])

    # get the mean of a students coursework
    averagecwk = pandas.DataFrame(stuAssessments.groupby("id_student")["score"].mean())
    averagecwk.reset_index(level=0, inplace=True)
    stuInfo = stuInfo.merge(averagecwk, how="outer", on=["id_student"])

    averagecwk = pandas.DataFrame(stuAssessments.groupby("id_student")["score"].sum()) \
        .reset_index(level=0).rename(columns={'score':'totalScore'})
    stuInfo = stuInfo.merge(averagecwk, how="outer", on=["id_student"])

    # get the mean score for a given person's daysEarly
    daysEarlyAvg = stuInfo.groupby("daysEarly")["score"].mean() \
        .reset_index(level=0).rename(columns={"score": "daysEarlyAvgScore"})
    stuInfo = stuInfo.merge(daysEarlyAvg, how="outer", on=["daysEarly"])

    # get the median score for a given person's daysEarly
    daysEarlyAvg = stuInfo.groupby("daysEarly")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "daysEarlyMedScore"})
    stuInfo = stuInfo.merge(daysEarlyAvg, how="outer", on=["daysEarly"])

    # get the mad score for a given person's daysEarly
    daysEarlyAvg = stuInfo.groupby("daysEarly")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "daysEarlyMadScore"})
    stuInfo = stuInfo.merge(daysEarlyAvg, how="outer", on=["daysEarly"])

    # get the var score for a given person's daysEarly
    daysEarlyAvg = stuInfo.groupby("daysEarly")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "daysEarlyvarScore"})
    stuInfo = stuInfo.merge(daysEarlyAvg, how="outer", on=["daysEarly"])

    # get the std score for a given person's daysEarly
    daysEarlyAvg = stuInfo.groupby("daysEarly")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "daysEarlystdScore"})
    stuInfo = stuInfo.merge(daysEarlyAvg, how="outer", on=["daysEarly"])
    # get the mean score for a given imd band
    imdAvg = stuInfo.groupby("imd_band")["score"].mean() \
        .reset_index(level=0).rename(columns={"score": "imd_bandAvgScore"})
    stuInfo = stuInfo.merge(imdAvg, how="outer", on=["imd_band"])

    # get the median score for a given imd band
    imdAvg = stuInfo.groupby("imd_band")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "imd_bandMedScore"})
    stuInfo = stuInfo.merge(imdAvg, how="outer", on=["imd_band"])

    # get the mad score for a given imd band
    imdAvg = stuInfo.groupby("imd_band")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "imd_bandMadScore"})
    stuInfo = stuInfo.merge(imdAvg, how="outer", on=["imd_band"])

    # get the var score for a given imd band
    imdAvg = stuInfo.groupby("imd_band")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "imd_bandvarScore"})
    stuInfo = stuInfo.merge(imdAvg, how="outer", on=["imd_band"])

    # get the std score for a given imd band
    imdAvg = stuInfo.groupby("imd_band")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "imd_bandstdScore"})
    stuInfo = stuInfo.merge(imdAvg, how="outer", on=["imd_band"])

    # get the mean score for a given age band
    ageAvg = stuInfo.groupby("age_band")["score"].mean() \
        .reset_index(level=0).rename(columns={"score": "age_bandAvgScore"})
    stuInfo = stuInfo.merge(ageAvg, how="outer", on=["age_band"])

    # get the median score for a given age band
    ageAvg = stuInfo.groupby("age_band")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "age_bandMedScore"})
    stuInfo = stuInfo.merge(ageAvg, how="outer", on=["age_band"])

    # get the mad score for a given age band
    ageAvg = stuInfo.groupby("age_band")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "age_bandMadScore"})
    stuInfo = stuInfo.merge(ageAvg, how="outer", on=["age_band"])

    # get the var score for a given age band
    ageAvg = stuInfo.groupby("age_band")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "age_bandvarScore"})
    stuInfo = stuInfo.merge(ageAvg, how="outer", on=["age_band"])

    # get the std score for a given age band
    ageAvg = stuInfo.groupby("age_band")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "age_bandstdScore"})
    stuInfo = stuInfo.merge(ageAvg, how="outer", on=["age_band"])

    # get the mean score for a given age band
    attemptsAvg = stuInfo.groupby("num_of_prev_attempts")["score"].mean() \
        .reset_index(level=0).rename(columns={"score": "num_of_prev_attemptsAvgScore"})
    stuInfo = stuInfo.merge(attemptsAvg, how="outer", on=["num_of_prev_attempts"])

    # get the median score for a given attempts
    attemptsAvg = stuInfo.groupby("num_of_prev_attempts")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "num_of_prev_attemptsMedScore"})
    stuInfo = stuInfo.merge(attemptsAvg, how="outer", on=["num_of_prev_attempts"])

    # get the mad score for a given attempts
    attemptsAvg = stuInfo.groupby("num_of_prev_attempts")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "num_of_prev_attemptsMadScore"})
    stuInfo = stuInfo.merge(attemptsAvg, how="outer", on=["num_of_prev_attempts"])

    # get the var score for a given attempts
    attemptsAvg = stuInfo.groupby("num_of_prev_attempts")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "num_of_prev_attemptsvarScore"})
    stuInfo = stuInfo.merge(attemptsAvg, how="outer", on=["num_of_prev_attempts"])

    # get the std score for a given attempts
    attemptsAvg = stuInfo.groupby("num_of_prev_attempts")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "num_of_prev_attemptsstdScore"})
    stuInfo = stuInfo.merge(attemptsAvg, how="outer", on=["num_of_prev_attempts"])

    # get the weighted score of the assessments ie the summative scores
    temp = (all_Assessments["score"] * all_Assessments["weight"]/100)
    all_Assessments.insert(loc=0, value=temp, column="weightedScore")

    summative = all_Assessments.groupby(["id_student","code_module", "code_presentation"])["weightedScore"].sum()
    summative = pandas.DataFrame(summative).rename(columns={"weightedScore": "summative"})
    summative.reset_index(level=0, inplace=True)

    # get the formative scores ie where the course weighting is 0
    temp = all_Assessments[["id_student","code_module", "code_presentation", "score"]].where(all_Assessments["weight"] == 0)

    formative = pandas.DataFrame(temp.groupby(["id_student","code_module", "code_presentation"]).mean()).rename(columns={"score": "formative"}).fillna(0)
    formative.reset_index(level=0, inplace=True)

    # get the mean score of a given assessment
    assessmentMeans = all_Assessments.groupby("id_assessment")["score"].mean() \
        .reset_index(level=0).rename(columns={"score": "assessment_Mean"})
    all_Assessments = all_Assessments.merge(assessmentMeans)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_Mean"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the median score of a given assessment
    assessmentMeds = all_Assessments.groupby("id_assessment")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "assessment_Meds"})
    all_Assessments = all_Assessments.merge(assessmentMeds)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_Meds"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the standard deviation of the score of a given assessment
    assessmentStd = all_Assessments.groupby("id_assessment")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "assessment_Std"})
    all_Assessments = all_Assessments.merge(assessmentStd)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_Std"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the mean absolute deviation of the score of a given assessment
    assessment_mad = all_Assessments.groupby("id_assessment")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "assessment_mad"})
    all_Assessments = all_Assessments.merge(assessment_mad)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_mad"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the variance of the score of a given assessment
    assessment_var = all_Assessments.groupby("id_assessment")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "assessment_var"})
    all_Assessments = all_Assessments.merge(assessment_var)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_var"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the mean assessment_type of a course
    assessment_type_mean = all_Assessments.groupby("id_assessment")["assessment_type"].mean() \
        .reset_index(level=0).rename(columns={"assessment_type": "assessment_type_mean"})
    all_Assessments = all_Assessments.merge(assessment_type_mean)
    TotalAssAverage = all_Assessments.groupby("id_student")["assessment_type_mean"].mean().reset_index(level=0)
    stuInfo = stuInfo.merge(TotalAssAverage, how="outer", on=["id_student"])

    # get the total number of coursework assignments a student attempted
    totalCwk = all_Assessments.groupby("id_student")["id_assessment"].count() \
        .reset_index(level=0).rename(columns={"id_assessment": "totalCoursework"})
    stuInfo = stuInfo.merge(totalCwk, how="outer", on=["id_student"])

    # get the median score of a given students assessments
    stuMeds = all_Assessments.groupby("id_student")["score"].median() \
        .reset_index(level=0).rename(columns={"score": "student_Meds"})
    stuInfo = stuInfo.merge(stuMeds, how="outer", on=["id_student"])

    # get the standard deviation of the score of a given students assessments
    stuSTD = all_Assessments.groupby("id_student")["score"].std() \
        .reset_index(level=0).rename(columns={"score": "student_Std"})
    stuInfo = stuInfo.merge(stuSTD, how="outer", on=["id_student"])

    # get the mean absolute deviation of the score of a given students assessments
    stuMad = all_Assessments.groupby("id_student")["score"].mad() \
        .reset_index(level=0).rename(columns={"score": "student_mad"})
    stuInfo = stuInfo.merge(stuMad, how="outer", on=["id_student"])

    # get the variance of the score of a given students assessments
    stuVar = all_Assessments.groupby("id_student")["score"].var() \
        .reset_index(level=0).rename(columns={"score": "student_var"})
    stuInfo = stuInfo.merge(stuVar, how="outer", on=["id_student"])

    # get the mean assessment_type of a course
    stu_type_mean = all_Assessments.groupby("id_student")["assessment_type"].mean() \
        .reset_index(level=0).rename(columns={"assessment_type": "stu_type_mean"})
    stuInfo = stuInfo.merge(stu_type_mean, how="outer", on=["id_student"])

    # merge this data into a new table called modelData

    modelData = stuInfo

    modelData = modelData.merge(summative, how="outer", on=["id_student","code_module", "code_presentation"])
    modelData = modelData.merge(formative, how="outer", on=["id_student","code_module", "code_presentation"])
    modelData = modelData.merge(visits, how="outer", on=["id_student"])

    # Fills those without formative scores with 0 rather than the median formative mark
    modelData['formative'] = modelData['formative'].fillna(0)

    # scale the summative score based on the number of credits they studied
    # otherwise they had a summative score over 100%
    modelData["summative"] = modelData["summative"] / modelData["studied_credits"]
    summative.rename(columns={"summative": "summativeAgainstCredits"}, inplace=True)
    modelData = modelData.merge(summative, how="outer", on=["id_student","code_module", "code_presentation"])

    # make final result categorical, order it then swap the column to only use to codes
    # as we can handle numbers and not strings
    modelData["final_result"] = modelData["final_result"].astype("category")
    modelData["final_result"] = modelData["final_result"].cat.reorder_categories(
        ["Withdrawn", "Fail", "Pass", "Distinction"], ordered=True)
    # 0, 1, 2, 3
    modelData["final_result"] = modelData["final_result"].cat.codes
    modelData["final-result2"] = modelData["final_result"].replace({3: 2, 0: 1})
    modelData["final-result3"] = modelData["final_result"].replace({3: 2})

    # make gender categorical, then swap the column to only use to codes
    modelData["gender"] = modelData["gender"].astype("category")
    modelData["gender"] = modelData["gender"].cat.codes
    # make disability categorical, then swap the column to only use to codes
    modelData["disability"] = modelData["disability"].astype("category")
    modelData["disability"] = modelData["disability"].cat.codes

    # make highest education categorical, order it based on quality of qualification
    # then swap the column to only use to codes
    modelData["highest_education"] = modelData["highest_education"].astype("category")
    modelData["highest_education"] = modelData["highest_education"].cat.reorder_categories(
        ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification",
         "Post Graduate Qualification"],
        ordered=True)
    modelData["highest_education"] = modelData["highest_education"].cat.codes

    modelData["id_student"] = modelData["id_student"].astype("category")
    modelData["code_module"] = modelData["code_module"].astype("category")
    modelData["code_presentation"] = modelData["code_presentation"].astype("category")
    modelData.columns = [x.replace("_","-") for x in np.array(modelData.columns,dtype=str)]
    return modelData


def preprocessingFunction(modelData):
    '''
    categorise region & impute / scale numerical data
    :param modelData: the gathered data for a given row in stuInfo
    :return: the processed modelData
    '''
    #####################---DATA Pre-Processing---#####################
    print(modelData.describe())
    # descriptions of the data gathered

    # generate the list of numerical and categorical attributes
    features = set(list(modelData.columns)).difference({"final-result","final-result2","final-result3","id-student"})
    num_attributes = features.difference({"region","code-module", "code-presentation"})
    cat_attributes = features.difference(num_attributes)
    num_attributes = list(num_attributes)
    cat_attributes = list(cat_attributes)

    # make a numerical pipeline by filling na values with the median
    # and scale everything to 0-1
    numericalPipeline = pipeline.Pipeline([
        ("imputer", impute.SimpleImputer(strategy="median")),
        ("min_max_scaler", preprocessing.MinMaxScaler())
    ])
    # run the data through the pipeline
    modelData[num_attributes] = numericalPipeline.fit_transform(modelData[num_attributes])

    # make columns specific to regions to spot regional patterns
    modelData = modelData.join(pandas.get_dummies(modelData[cat_attributes]))

    modelData = modelData.drop(columns=cat_attributes)

    print("Finished data gather - Time elapsed {:.2f}".format(time.time() - start))
    # output the correlations & heatmap
    modelData.columns = [x.replace("_","-") for x in np.array(modelData.columns,dtype=str)]
    corr = modelData.corr()
    corr = corr.reindex(corr['final-result'].sort_values()
                        .reset_index(level=0).iloc[:,0],axis=1).sort_values(by=['final-result'])
    with pandas.option_context('display.max_rows',None,'display.max_columns',None):
        print(corr["final-result"])

    # produce & save heatmap based on correlations
    sns.set(font_scale=0.32)
    plt.figure()

    heatmap = sns.heatmap(corr, cmap="YlOrRd",
                          xticklabels=2,
                          yticklabels=1,
                          annot=False, cbar_kws={'label': 'Correlation'})
    plt.xlabel('Features',fontsize=10)
    plt.title('Correlation Heatmap', fontsize=17)
    figure = heatmap.get_figure()
    saveFigure("heatmap")

    return modelData

# run above functions
modelData = dataGather()
modelData = preprocessingFunction(modelData)

# split the model data into test and training data
train, test_final = model_selection.train_test_split(modelData, random_state=random_state, test_size=0.25)

#####################---Model Selection---#####################

# Define features to use for fitting
attributes = list(modelData.columns)
attributes.remove("final-result")
attributes.remove("final-result2")
attributes.remove("final-result3")
regreAttributes = attributes.copy()
regreAttributes.remove('id-student')

# initialise regression & classification objects

# set logistic regression to use multinomial class
# have a max iterations of 10000 (struggles to find pattern otherwise)
logRegr = linear_model.LogisticRegression(multi_class='multinomial', max_iter=100000)
linRegr = linear_model.LinearRegression()
decClass = tree.DecisionTreeClassifier()
forClass = ensemble.RandomForestClassifier( class_weight='balanced')
svrRegr = svm.SVR(kernel="linear")
svrPolyRegr = svm.SVR(kernel="poly")
svrRBF = svm.SVR(kernel="rbf")
svmC = svm.SVC(class_weight='balanced')


def modelSelectionScore(model,features, Name="", classes=4):
    '''
    Run cross val score on the model provided, printing the findings
    :param model: model instance ie logisticRegression or RandomForestClassifier
    :param features: which feature set to use (ie regreAttributes or attributes - model dependant)
    :param Name: Any add on to the name provided through the model's class
    :param classes: How many classes to train the data on
    '''
    print("%s: " % (
            model.__class__.__name__ + Name))
    scores = None
    if classes == 4:
        scores = model_selection.cross_val_score(model, train[features], train["final-result"], cv=5)
    elif classes == 3:
        scores = model_selection.cross_val_score(model, train[features], train["final-result3"], cv=5)
    elif classes == 2:
        scores = model_selection.cross_val_score(model, train[features], train["final-result2"], cv=5)
    else:
        exit()

    print("Accuracy:  " + str(scores))
    print("Mean Accuracy Across folds:  " + str(scores.mean()))
    print("Standard Deviation of the Accuracy Across folds:  " + str(scores.std()))
    print("\n")


def plots(model, subtrain, subtest, feature, Name="", logistic=False):
    '''

    :param model: model to plot againt
    :param subtrain: subtrain split
    :param subtest: subtest split
    :param feature: which feature to train and plot against
    :param Name: Name of the model
    :param logistic: boolean for non linear plots - using probabilities instead
    '''
    print("\n{name} model against {featureName}".format(name=model.__class__.__name__+Name,
                                                        featureName=feature.replace("_", " ")))
    model.fit(np.array(subtrain[feature]).reshape(-1, 1), np.array(subtrain["final-result2"]).ravel())

    # Make predictions using the testing set
    featurePred = model.predict(np.array(subtest[feature]).reshape(-1, 1))

    try:
        # The coefficients
        print('Coefficients: \n', model.coef_)
    except AttributeError:
        pass
    # The mean squared error
    print('Mean squared error: %.2f'
          % metrics.mean_squared_error(subtest['final-result2'], featurePred))
    # The r2 accuracy
    print('r2 Accuracy Score: %.2f'
          % metrics.r2_score(subtest["final-result2"], featurePred))
    # The score - unique to model
    print('Score: %.2f'
          % model.score(np.array(subtrain[feature]).reshape(-1, 1), np.array(subtrain["final-result2"]).ravel()))
    # if not linear plot the probability
    if logistic:
        featurePred = model.predict_proba(np.array(subtest[feature]).reshape(-1, 1))[:, 1] + 1

    featurePred.sort()
    plt.plot(subtest[feature].sort_values(), featurePred, color='blue', linewidth=1)
    plotSave("{name} 2 class model against {featureName}".format(
        name=model.__class__.__name__ + Name, featureName=feature.replace("_", " ")),
        subtest[feature], feature.replace("_", " "), subtest["final-result2"], "final result - 2 class")



def modelSelection():
    # generate tests to see how well each regression model performs without any hyper tuning
    # this is done using cross val score, with 10 folds,
    # ie it splits the data 10 times, using a different one of those subsets as a validation set and
    # returns the scores of how well the model performed

    subtrain, subtest = model_selection.train_test_split(train, random_state=random_state, test_size=0.2)

    modelSelectionScore(linRegr,regreAttributes, classes=4)

    # Linear Regr plots ##################################################################################
    # graph 4 data points for linear regression to compare

    plots(linRegr, subtrain, subtest, "sum-click")

    plots(linRegr, subtrain, subtest, "score")

    plots(linRegr, subtrain, subtest, "age-band")

    plots(linRegr, subtrain, subtest, "imd-band")

    # Logistic
    modelSelectionScore(logRegr,regreAttributes,Name=" - 4 classes", classes=4)
    modelSelectionScore(logRegr,regreAttributes,Name=" - 3 classes", classes=3)
    modelSelectionScore(logRegr,regreAttributes,Name=" - 2 classes", classes=2)

    plots(logRegr, subtrain, subtest, "score", logistic=True)
    plots(logRegr, subtrain, subtest, "summative", logistic=True)
    # SVR Linear
    modelSelectionScore(svrRegr,regreAttributes, " - Linear Kernel")

    # SVR Linear plots ##################################################################################
    # graph 3 data points for svr linear regression to compare

    plots(svrRegr, subtrain, subtest, "sum-click", "-Linear-Kernel")

    plots(svrRegr, subtrain, subtest, "score", "-Linear-Kernel")

    plots(svrRegr, subtrain, subtest, "imd-band", "-Linear-Kernel")

    # SVR Poly
    modelSelectionScore(svrPolyRegr,regreAttributes, " - Polynomial Kernel")

    # SVR poly plots ##################################################################################
    # graph 3 data points for svr linear regression to compare

    plots(svrPolyRegr, subtrain, subtest, "sum-click", "-Polynomial-Kernel")

    plots(svrPolyRegr, subtrain, subtest, "score", "-Polynomial-Kernel")

    plots(svrPolyRegr, subtrain, subtest, "age-band", "-Polynomial-Kernel")

    # SVR RBF (Current Sklearn Default)
    modelSelectionScore(svrRBF,regreAttributes, " - RBF Kernel")

    # SVM Classifier
    modelSelectionScore(svmC,regreAttributes," - 4 classes",4)
    modelSelectionScore(svmC,regreAttributes," - 3 classes",3)
    modelSelectionScore(svmC,regreAttributes," - 2 classes",2)

    # Decision Tree Classifier
    modelSelectionScore(decClass,attributes," - 4 classes",4)
    modelSelectionScore(decClass,attributes, " - 3 classes", 3)
    modelSelectionScore(decClass,attributes, " - 2 classes", 2)

    # Random Forest Classifier
    modelSelectionScore(forClass,attributes," - 4 classes",4)
    modelSelectionScore(forClass,attributes, " - 3 classes", 3)
    modelSelectionScore(forClass,attributes, " - 2 classes", 2)

    print("Finished Model Selection - Time elapsed {:.2f}".format(time.time() - start))


# -> compare logistic regression and random forest
# -> fine tune

def hypertuning(modelA, modelB):
    '''
    Tunes model A with grid search and model B with bayesian optimisation
    :param modelA: boolean for running model A - primarily for testing
    :param modelB: boolean for running model B - primarily for testing
    :return: Outputs the final metrics for evaluating the two models
    '''
    #####################---Fine Tune---#####################
    print("\n\nHyper Parameter Tuning:")

    if modelA:
        # perform a grid search for the logistic regression,
        # by comparing performance of each combination for the following parameters
        # therefore checks 6 * 5 = 30 combinations of results
        # uses the cross validation steps used earlier with 5 folds
        param_grid = [
            # {"C": range(950,1125,25), "penalty": },
            {"C": np.logspace(-3,3.5,10), "penalty":['l1','l2']},
            # {"C": [10], "tol": [0.00016]},
        ]

        LogGridSearch = model_selection.GridSearchCV(logRegr, param_grid, cv=5, return_train_score=True, n_jobs=-1,
                                                     verbose=False)
        logSearch = LogGridSearch.fit(train[regreAttributes], train["final-result"])

        # the best parameters found are stored in best params
        print("The Logistic Regression Model produced a mean best score of {:5.3f}% in Hyper Tuning".format(
            logSearch.best_score_ * 100))
        print("The parameters which produced this score are {:}".format(logSearch.best_params_))
        print("Finished modelA hyper parameter tuning - Time elapsed {:.2f}".format(time.time() - start))

    if modelB:

        def objectiveA(params):
            '''
            objective function for all features
            :param params: Random Forest Parameters for the current iteration
            :return: Dictionary containing, the loss, the params which produced the loss & a status
            '''

            forClass.set_params(**params)
            results = model_selection.cross_validate(forClass, train[attributes], train["final-result"], cv=5,
                                                      n_jobs=-1, scoring=["accuracy","f1_weighted",
                                                                          "balanced_accuracy"])

            loss = 3-results["test_accuracy"].mean() - results["test_f1_weighted"].mean() - \
            results["test_balanced_accuracy"].mean()

            return {"loss":loss, 'params': params, 'status': hyperopt.STATUS_OK,'acc':results["test_accuracy"].mean(),'f1':results["test_f1_weighted"].mean(),'balAcc': results["test_balanced_accuracy"].mean()}
        def objectiveB(params):
            '''
            objective function for only important features
            :param params: Random Forest Parameters for the current iteration
            :return: Dictionary containing, the loss, the params which produced the loss & a status
            '''

            forClass.set_params(**params)
            results = model_selection.cross_validate(forClass, train[importantFeatures], train["final-result"], cv=5,
                                                      n_jobs=-1, scoring=["accuracy","f1_weighted",
                                                                          "balanced_accuracy"])

            loss = 3-results["test_accuracy"].mean() - results["test_f1_weighted"].mean() - \
            results["test_balanced_accuracy"].mean()
            return {"loss":loss, 'params': params, 'status': hyperopt.STATUS_OK,'acc':results["test_accuracy"].mean(),'f1':results["test_f1_weighted"].mean(),'balAcc': results["test_balanced_accuracy"].mean()}

        # Iterations of hypertuning
        max_evals = 1000

        # search space for the hyperparameter optimisation search problem
        param_space = {"n_estimators":hyperopt.hp.choice("n_estimators", np.arange(1, 1000, dtype=int)),
                        "max_depth": hyperopt.hp.choice("max_depth",np.arange(1, 320, dtype=int)),
                        "min_samples_leaf": hyperopt.hp.choice("min_samples_leaf",np.arange(1, 320, dtype=int)),
                        "min_samples_split": hyperopt.hp.choice("min_samples_split",np.arange(2, 100, dtype=int)),
                        "class_weight":hyperopt.hp.choice("class_weight",[
                            None,{0:hyperopt.hp.uniform("0",0,10),1:hyperopt.hp.uniform("1",0,10), 2:
                                             hyperopt.hp.uniform("2",0,10), 3:hyperopt.hp.uniform("3",0,10)}
                            # removed balanced in order for hyperopt to optimise the numerical weighting
                         ])
                        }

        # trials - hyperopt object storing information about each iterations
        # like the dictionary returned from the objective function
        trials = hyperopt.Trials()
        # minimisation function - returns the best parameters found
        best = hyperopt.fmin(fn=objectiveA,space=param_space,algo=hyperopt.tpe.suggest, max_evals=int(max_evals*4/5),
                             trials=trials)
        print(trials.best_trial)
        # converts from array indices to
        best = hyperopt.space_eval(param_space,best)
        forClass.set_params(**best)

        forClass.fit(train[attributes],train["final-result"])

        print("The Random Forest Classifier Model produced a best loss of {:5.3f}% in Hyper Tuning".format(
            (trials.best_trial["result"]["loss"]) * 100))
        print("The parameters which produced this score are {:}".format(best))
        print("Finished modelB hyper parameter tuning round 1 - Time elapsed {:.2f}".format(time.time() - start))

        MM = plt.scatter(range(len(trials)),[3-i["result"]["loss"] for i in trials.trials],alpha=0.2)
        Loss = plt.scatter(range(len(trials)), [i["result"]["loss"] for i in trials.trials], alpha=0.2)
        Acc = plt.scatter(range(len(trials)), [i["result"]["acc"] for i in trials.trials], alpha=0.2)
        balAcc = plt.scatter(range(len(trials)), [i["result"]["balAcc"] for i in trials.trials], alpha=0.2)
        f1 = plt.scatter(range(len(trials)), [i["result"]["f1"] for i in trials.trials], alpha=0.2)
        plt.legend((MM,Loss,Acc,balAcc,f1),('Summed Accuracy, Balanced Accuracy & f1 means','Loss','Mean Accuracy', 'Mean Balanced Accuracy','Mean f1'),loc='right')
        plt.xlabel("Iterations",fontsize=10)
        # plt.title("Metrics against iteration",fontsize=17)
        saveFigure("Accuracy_against_iteration1")



        print("Removing unimportant features and tuning again")
        # Get numerical feature importance
        importances = list(forClass.feature_importances_)
        feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(attributes, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

        [print('Variable: {:35} Importance: {}'.format(*pair)) for pair in feature_importances]

        sorted_importances = [importance[1] for importance in feature_importances]
        sorted_features = [importance[0] for importance in feature_importances]
        cumulative_importances = np.cumsum(sorted_importances)
        plt.plot(list(range(len(importances))), cumulative_importances)
        # add the x labels
        plt.xticks(list(range(len(importances))), sorted_features, rotation='vertical')
        # plot horizontal line across importance cut off point
        plt.hlines(0.95,0,len(importances),lw=1)
        plt.xlabel('Feature',fontsize=10)
        plt.ylabel('Cumulative Importance',fontsize=10)
        # plt.title('Cumulative Importances of features',fontsize=17)
        saveFigure("Importances")

        # assigns important features to top 95 % of importance
        sizeFeatures = np.where(cumulative_importances > 0.95)[0][0]
        importantFeatures = sorted_features[:sizeFeatures]

        # trials - hyperopt object storing information about each iterations
        # like the dictionary returned from the objective function
        trials = hyperopt.Trials()
        # minimisation function - returns the best parameters found
        best = hyperopt.fmin(fn=objectiveB, space=param_space, algo=hyperopt.tpe.suggest, max_evals=max_evals,
                      trials=trials)

        # plot loss & mean metric against iteration
        MM = plt.scatter(range(len(trials)),[3-i["result"]["loss"] for i in trials.trials],alpha=0.2, color='r')
        Loss = plt.scatter(range(len(trials)), [i["result"]["loss"] for i in trials.trials], alpha=0.2, color='b')
        Acc = plt.scatter(range(len(trials)), [i["result"]["acc"] for i in trials.trials], alpha=0.2)
        balAcc = plt.scatter(range(len(trials)), [i["result"]["balAcc"] for i in trials.trials], alpha=0.2)
        f1 = plt.scatter(range(len(trials)), [i["result"]["f1"] for i in trials.trials], alpha=0.2)
        plt.legend((MM,Loss,Acc,balAcc,f1),('Summed Accuracy, Balanced Accuracy & f1 means','Loss','Mean Accuracy', 'Mean Balanced Accuracy','Mean f1'),loc='right')
        plt.xlabel("Iterations",fontsize=10)
        # plt.title("Metrics against iteration",fontsize=17)
        saveFigure("Accuracy_against_iteration2")

        # plot estimators against iteration
        plt.scatter(range(len(trials)),[i["result"]["params"]['n_estimators'] for i in trials.trials],alpha=0.2)
        plt.xlabel("Iterations",fontsize=10)
        plt.ylabel("N_Estimators",fontsize=10)
        # plt.title("Estimators against iteration",fontsize=17)
        saveFigure("Estimators_against_iteration")

        # convert from domain to specific values
        best = hyperopt.space_eval(param_space,best)
        forClass.set_params(**best)

        forClass.fit(train[importantFeatures], train["final-result"])

        print("The Random Forest Classifier Model produced a best loss of {:5.3f}% in Hyper Tuning".format(
            (trials.best_trial["result"]["loss"]) * 100))
        print("The parameters which produced this score are {:}".format(best))
        print("Finished modelB hyper parameter tuning round 2 - Time elapsed {:.2f}".format(time.time() - start))

    #####################---Final Analysis---#####################

    if modelA:
        # run the final analysis
        # use the model found from hyper tuning
        logisticModel = logSearch.best_estimator_

        # make the prediction
        logisticPrediction = logisticModel.predict(test_final[regreAttributes])

        # scoring tuple
        truePredTuple = (test_final["final-result"], logisticPrediction)

        # output all the scorings for the logistic regression model
        print("\n\nFinal Logistic Regression Model - 4 Class:")
        finalMetrics(truePredTuple)

        # Evaluation of 2 class version of model using parameters found from tuning
        logisticModel = logSearch.best_estimator_
        logisticModel.fit(test_final[regreAttributes],test_final["final-result2"])
        # make the prediction
        logisticPrediction = logisticModel.predict(test_final[regreAttributes])

        # scoring tuple
        truePredTuple = (test_final["final-result2"], logisticPrediction)

        # output all the scorings for the logistic regression model
        print("\n\nFinal Logistic Regression Model - 2 Class:")
        finalMetrics(truePredTuple)

    if modelB:
        # use the model found from hyper tuning
        randomForestModel = forClass

        # make prediction
        randomForestPrediction = randomForestModel.predict(test_final[importantFeatures])

        # scoring tuple
        truePredTuple = (test_final["final-result"], randomForestPrediction)

        # output metrics for the random forest regression model
        print("\n\nFinal Random Forest Classifier Model - 4 Class:")
        finalMetrics(truePredTuple)


        # 2 class problem
        # Evaluation of 2 class version of model using parameters found from tuning
        randomForestModel = forClass
        # class weight as 4 dictionary fails for 2 classes - so balanced is used instead
        best["class_weight"] = "balanced"
        randomForestModel.set_params(**best)

        randomForestModel.fit(test_final[importantFeatures],test_final["final-result2"])

        # make prediction
        randomForestPrediction = randomForestModel.predict(test_final[importantFeatures])

        # scoring tuple
        truePredTuple = (test_final["final-result2"], randomForestPrediction)

        # output all the scorings for the random forest regression model
        print("\n\nFinal Random Forest Classifier Model - 2 Class:")
        finalMetrics(truePredTuple)


    print("Finished - Time elapsed {:.2f}".format(time.time() - start))


def finalMetrics(truePredTuple):
    print("Explained Variance Score: %.3f" % metrics.explained_variance_score(truePredTuple[0], truePredTuple[1]))
    print("Mean Absolute Error: %.3f" % metrics.mean_absolute_error(truePredTuple[0], truePredTuple[1]))
    print("Mean Square Error: %.3f" % metrics.mean_squared_error(truePredTuple[0], truePredTuple[1]))
    print("Root Mean Square Error: %.3f" % metrics.mean_squared_error(truePredTuple[0], truePredTuple[1],
                                                                      squared=False))
    print("r2 Score (Accuracy): %.3f" % metrics.r2_score(truePredTuple[0], truePredTuple[1]))
    print("Accuracy Score: %.3f" % metrics.accuracy_score(truePredTuple[0], truePredTuple[1]))
    category2String = {0:'Withdrawn',1: "Fail", 2: "Pass",3:'Distinction'}

    print(metrics.classification_report([category2String[i] for i in truePredTuple[0]],
                                        [category2String[i] for i in truePredTuple[1]], digits=3))
    if len(truePredTuple[0].unique()) == 2:
        category2String = ["Fail", "Pass"]
    else:
        category2String = ["Withdrawn", "Fail", "Pass", "Distinction"]

    print(pandas.DataFrame(metrics.confusion_matrix(truePredTuple[0], truePredTuple[1]), index=category2String,
                           columns=category2String))

# modelSelection()
hypertuning(True, True)
