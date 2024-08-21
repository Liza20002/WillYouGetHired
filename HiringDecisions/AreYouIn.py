#In this file we will do our statistical research. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import norm
from scipy import stats
from scipy.stats import chi2_contingency








HiringData = pd.read_csv("Data/recruitment_data.csv")
#our statistical significance level is alpha = 0.05
alpha = 0.05

#################################
#Question 1
#################################
#We want to find out the connetction between the skills level of the candidate and their acceptence rate.
#Our H0 - base assumption is that the average skill score of candidates that got hired is 70
mu_0 = 70

#Here we will only take those who were accepted
accepted_candidates = HiringData[HiringData['HiringDecision'] == 1]


#Finding the amount of accepted candidates
num_of_accepted_cands = len(accepted_candidates)

#Calculating the unbiased variance of the SkillScore column:
unbiased_variance = np.var(accepted_candidates['SkillScore'], ddof=1)


#Now we will find the Standard deviation of the accepted candidates
std_dev_skill_score = np.sqrt(unbiased_variance)


#Estimating the mean SkillScore of accepted candidates:
mean_skill_score_accepted = accepted_candidates['SkillScore'].mean()

#Here we will show that we can calculate the 't statistic' on our own, and later we will do that using an imported function
t_statistic_skill = (mean_skill_score_accepted - mu_0) / (std_dev_skill_score/np.sqrt(num_of_accepted_cands))


#Calculate the critical value for a two-tailed test
critical_value = t.ppf(1 - alpha/2, df=num_of_accepted_cands - 1) 

#Calculating the p-value for two tailed
#We can see that they use CDF to find the values that are equal or greater(or smaller) then the t- stat we found
p_value = 2 * (1 - t.cdf(abs(t_statistic_skill), num_of_accepted_cands - 1)) 

#The t-statistic is lower the the critical value t_statistic<critical_value so we reject the HO
#################3 finished two tailed test

#We'll assume that you need to have 60 in personality to get accepted. Our new mu_0 is <=60,
mu_0 = 60
#Our H1 is that mu is > 60 


#Calculating the t statistic and p-value using the imported library:
t_statistic_personality, p_value = ttest_1samp(accepted_candidates['PersonalityScore'], mu_0)

critical_value = t.ppf(1 - alpha, num_of_accepted_cands - 1) 

#We found that out t_statistic < critical value, so we accept H0

##################


#Now our H0 for mu of distance of accepted candidates is 15 KM or more
mu_0 = 15
#Our H1 is that if distance is less then 15 KM we reject it
#Calculating the t statistic and p- value using the imported library:
t_statistic, p_value = ttest_1samp(accepted_candidates['DistanceFromCompany'], mu_0)
#Calculating the critical value:
critical_value = -t.ppf(1 - alpha, num_of_accepted_cands - 1) 
#We found that our t_Statistic is bigger than the critical value, so we accept H0


#################################
# Question 2
#################################

#hypothesis testing for variance 
#we'll check if the skill score has to do with experiance in years
#we'll begin by taking the group of those who have over 13 years of experience 
very_experianced = HiringData[HiringData['ExperienceYears']>13]
amount_of_very_experianced = len(very_experianced)
#we have 209 candidates in this group
#we think that the skill score for this group is very high,and that the variance is not going to be big.
#first we will find the variance in SkillScore for the whole sample
variance_whole_sample = np.var(HiringData['SkillScore'], ddof=1)
#we know that the variance in SkillScore for the whole sample is 861.63
#our assumption is that because we have a group of people that are very expirienced,they are going to get very high skill score with a smaller variance than the whole test
# our H0 for the variance of this group is <= 750, and we will check that with right sided test
var_h0 = 750
#we will reject H0 if the chi value of the estimation will be greater then the chi value of our critical value
critical_value = chi2.ppf(1 - alpha, amount_of_very_experianced-1)
#the critical value is 242.64
#we will find an estimator for the mean of their skill set
mean_skill_experianced = very_experianced['SkillScore'].mean()
#we found that the mean skill score of this group is 51.976(surprisingly low)
#now let's find the variance of the sample
unbiased_variance_for_very_experianced = np.var(very_experianced['SkillScore'], ddof=1)
#our unbiased_variance_for_very_experianced is 756.56
#now we will find the value of our chi stat for variance, assuming the our H0 holds true
chi_stat_var = (amount_of_very_experianced-1) * unbiased_variance_for_very_experianced / var_h0 
#the chi value of our sample is 240.57, lower then the critical value of 242.64 ,so we accept our H0 with significance level "Ramat Muvhakut" of 0.05
print(chi_stat_var)
#let's find the P-value
p_value = 1- chi2.cdf(chi_stat_var,amount_of_very_experianced-1)

#the p_value is 1.0




#################################
# Question 3
#################################

#calculating the proportion of accepted candidates in all of the sample: 
proportion_accepted = HiringData['HiringDecision'].mean()

#we found that 0.31 were accepted
#we are going to check if there is a difference between people in different ages
#using discritization we will split into age groups of 5(20-25,25-30...)
#our H0 for acceptence rate of the group 25-30 is higher than the rest, because they are at the begining of their career, after completing their degree
p0 = 0.31
#our H1 is that the acceptence rate will be lower
#define the age bins (starting from 20 and ending at 50)
bins = [20, 25, 30, 35, 40, 45, 50] 
labels = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
HiringData['AgeGroup'] = pd.cut(HiringData['Age'], bins=bins,labels = labels, right=False)
only_25_30_candidates = HiringData[HiringData['AgeGroup'] == '25-29']
proportion_of_25_30_accepted = only_25_30_candidates['HiringDecision'].mean()
number_of_25_30_candidates = len(only_25_30_candidates)
#we found that the female acceptence rate is 0.308 in our sample
#we want to calculate the Z value of the proportion we found
z_proportion = (proportion_of_25_30_accepted - p0)/ np.sqrt(p0*(1-p0)/number_of_25_30_candidates)
#now we will use the normal distribution table for finding the z value of our significance level(right tailed test)
z_critical_bottom =-norm.ppf(1-alpha)
#now we will find the p-value for our test 
p_value_right_tailed = (norm.cdf(z_proportion))

#we found that our p-value is 0.957, which is bigger than our significance level 0.05, so we fail to reject H0

#################################
#Question 4
#################################

#In this question we will determine if there is a connection between the recruitment strategy(categorial betweewn 1-3) to the personality score the candidate recieved.
#first we will check the mean personallity grades in the entire sample
mean_personallity_score = HiringData['PersonalityScore'].mean()
#we found that the mean personality score in the sample is 49.387
#our H0 is that there is a connection and that when the interview is 'aggresive', the personallity score is lower.
#so we assume that the mu0 in aggresive interviews is going to be 45
mu0 = 45
#we will divide them by the strategy and check the personality score they got
agrresive_recruitment_strategy = HiringData[HiringData['RecruitmentStrategy']==1]
moderate_recruitment_strategy = HiringData[HiringData['RecruitmentStrategy']==2]
conservative_recruitment_strategy = HiringData[HiringData['RecruitmentStrategy']==3]

#let's find the mean of personality scores within the agressive recruitment group
mean_personallity_agrresive_recruitment_strategy = agrresive_recruitment_strategy['PersonalityScore'].mean() 
#we found mean of 49.676 (Higher than General score of the sample!!! surprising!!!)
#let's find an estimate to the variance of this group
unbiased_variance_personallity_when_agressive = np.var(agrresive_recruitment_strategy['PersonalityScore'], ddof=1)
std_dev_personallity_when_agressive = np.sqrt(unbiased_variance_personallity_when_agressive)
#let's find the number of people that faced agressive recruitment
amount_of_agressive_recruitmnent = len(agrresive_recruitment_strategy)

#now let's find the z-value of the statistic
t_statistic = (mean_personallity_agrresive_recruitment_strategy - mu0) / (std_dev_personallity_when_agressive/np.sqrt(amount_of_agressive_recruitmnent))
#we found that the z_statistic is 3.33
#the critical values are now tested using the T distribution
t_critical_top =t.ppf(1-alpha/2 ,amount_of_agressive_recruitmnent-1 )
t_critical_bottom = -t_critical_top
#we found that the critical value is 1.965 which is lower then the t_statistic that we found, so we reject our H0, the Recruitment Strategy does not significantly affect the personality score
#let's find the p-value (getting the result we got or bigger/smaller)
p_value_two_tailed = 2 * (1-t.cdf(abs(t_statistic) ,amount_of_agressive_recruitmnent-1))
#our p-value is 0.00009, which is smaller then our significance level, so we reject H0


#################################
# Question 5
#################################
#After we saw the graph of the distribution of personality score in Question 2, we suspect that the distribution is uniform.
#we will now check if the personality score is uniformally distrebiuted using hypothesis test
#our alpha stays 0.05, our H0 is that it does distribute uniformally
#our H1 is that it doesn't distribute uniformally

personallity_scores = HiringData['PersonalityScore']
#making a list of the observed frequncies of each personality score to later check if it distributes uniformally:
observed_frequencies = personallity_scores.value_counts()
observed_frequencies_list = observed_frequencies.values.tolist()
#the expected frequncies is the amount of all the scores divided by the amounts of each score (H0: distributes uniformmally)
expected_freq = len(personallity_scores) / len(observed_frequencies_list)
chi2_stat, p_value = stats.chisquare(f_obs=observed_frequencies_list, f_exp=[expected_freq] * len(observed_frequencies_list))
#we found that our p_value is 0.615, which is bigger than our alpha 0.05 so, we fail to reject H0 - the personallity scores distribute uniformmally.


#################################
#Question 6
#################################

#Our H0 is that the skill score is connected to the interview score, and has a significant affect on it
#we will divide the scores to groups of 10 (80-90, 90-100...)
bins = [0,10,20,30,40,50,60,70,80,90,100]
#here we will discritisize skill score and interview score using the groups we created before
HiringData['SkillScore_Discretized'] = pd.cut(HiringData['SkillScore'], bins=bins, labels=False)
HiringData['InterviewScore_Discretized'] = pd.cut(HiringData['InterviewScore'], bins=bins, labels=False)
#now we'll test using chi 2 test on contigency table with the new columns
contingency_table = pd.crosstab(HiringData['SkillScore_Discretized'], HiringData['InterviewScore_Discretized'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
#we found that the P-value is 0.498, which is way greater than 0.05, so we fail to reject the H0, there is no strong connection between the two parameters
#here we will show the results in a heat map
contingency_table = pd.crosstab(HiringData['SkillScore_Discretized'], HiringData['InterviewScore_Discretized'])

#Creating heatmap:
plt.figure(figsize=(10, 8))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Count'})
plt.title('Relationship Between Skill Score and Interview Score')
plt.xlabel('Interview Score (Discretized)')
plt.ylabel('Skill Score (Discretized)')
plt.show()
















 





















