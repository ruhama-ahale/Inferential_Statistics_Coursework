
# coding: utf-8

# # Hospital Readmissions Data Analysis and Recommendations for Reduction
# 
# ### Background
# In October 2012, the US government's Center for Medicare and Medicaid Services (CMS) began reducing Medicare payments for Inpatient Prospective Payment System hospitals with excess readmissions. Excess readmissions are measured by a ratio, by dividing a hospital’s number of “predicted” 30-day readmissions for heart attack, heart failure, and pneumonia by the number that would be “expected,” based on an average hospital with similar patients. A ratio greater than 1 indicates excess readmissions.
# 
# ### Exercise Directions
# 
# In this exercise, you will:
# + critique a preliminary analysis of readmissions data and recommendations (provided below) for reducing the readmissions rate
# + construct a statistically sound analysis and make recommendations of your own 
# 
# More instructions provided below. Include your work **in this notebook and submit to your Github account**. 
# 
# ### Resources
# + Data source: https://data.medicare.gov/Hospital-Compare/Hospital-Readmission-Reduction/9n3s-kdb3
# + More information: http://www.cms.gov/Medicare/medicare-fee-for-service-payment/acuteinpatientPPS/readmissions-reduction-program.html
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# ****

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bkp
from mpl_toolkits.axes_grid1 import make_axes_locatable
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')
import scipy.stats as stats
import seaborn

from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[3]:

# read in readmissions data provided
hospital_read_df = pd.read_csv('/Users/ruhama.ahale/Documents/Springboard_Coursework/hospital_readmit/cms_hospital_readmissions.csv')


# ****
# ## Preliminary Analysis

# In[4]:

# deal with missing and inconvenient portions of data 
clean_hospital_read_df = hospital_read_df[hospital_read_df['Number of Discharges'] != 'Not Available']
clean_hospital_read_df.loc[:, 'Number of Discharges'] = clean_hospital_read_df['Number of Discharges'].astype(int)
clean_hospital_read_df = clean_hospital_read_df.sort_values('Number of Discharges')


# In[5]:

# generate a scatterplot for number of discharges vs. excess rate of readmissions
# lists work better with matplotlib scatterplot function
x = [a for a in clean_hospital_read_df['Number of Discharges'][81:-3]]
y = list(clean_hospital_read_df['Excess Readmission Ratio'][81:-3])

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x, y,alpha=0.2)

ax.fill_between([0,350], 1.15, 2, facecolor='red', alpha = .15, interpolate=True)
ax.fill_between([800,2500], .5, .95, facecolor='green', alpha = .15, interpolate=True)

ax.set_xlim([0, max(x)])
ax.set_xlabel('Number of discharges', fontsize=12)
ax.set_ylabel('Excess rate of readmissions', fontsize=12)
ax.set_title('Scatterplot of number of discharges vs. excess rate of readmissions', fontsize=14)

ax.grid(True)
fig.tight_layout()


# ****
# 
# ## Preliminary Report
# 
# Read the following results/report. While you are reading it, think about if the conclusions are correct, incorrect, misleading or unfounded. Think about what you would change or what additional analyses you would perform.
# 
# **A. Initial observations based on the plot above**
# + Overall, rate of readmissions is trending down with increasing number of discharges
# + With lower number of discharges, there is a greater incidence of excess rate of readmissions (area shaded red)
# + With higher number of discharges, there is a greater incidence of lower rates of readmissions (area shaded green) 
# 
# **B. Statistics**
# + In hospitals/facilities with number of discharges < 100, mean excess readmission rate is 1.023 and 63% have excess readmission rate greater than 1 
# + In hospitals/facilities with number of discharges > 1000, mean excess readmission rate is 0.978 and 44% have excess readmission rate greater than 1 
# 
# **C. Conclusions**
# + There is a significant correlation between hospital capacity (number of discharges) and readmission rates. 
# + Smaller hospitals/facilities may be lacking necessary resources to ensure quality care and prevent complications that lead to readmissions.
# 
# **D. Regulatory policy recommendations**
# + Hospitals/facilties with small capacity (< 300) should be required to demonstrate upgraded resource allocation for quality care to continue operation.
# + Directives and incentives should be provided for consolidation of hospitals and facilities to have a smaller number of them with higher capacity and number of discharges.

# ****
# <div class="span5 alert alert-info">
# ### Exercise
# 
# Include your work on the following **in this notebook and submit to your Github account**. 
# 
# A. Do you agree with the above analysis and recommendations? Why or why not?
#    
# B. Provide support for your arguments and your own recommendations with a statistically sound analysis:
# 
#    1. Setup an appropriate hypothesis test.
#    2. Compute and report the observed significance value (or p-value).
#    3. Report statistical significance for $\alpha$ = .01. 
#    4. Discuss statistical significance and practical significance. Do they differ here? How does this change your recommendation to the client?
#    5. Look at the scatterplot above. 
#       - What are the advantages and disadvantages of using this plot to convey information?
#       - Construct another plot that conveys the same information in a more direct manner.
# 
# 
# 
# You can compose in notebook cells using Markdown: 
# + In the control panel at the top, choose Cell > Cell Type > Markdown
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# </div>
# ****

# In[6]:

# Your turn


# In[4]:

#Let us first check out the dataset and clean it of missing values
len(hospital_read_df)


# In[5]:

hospital_read_df.describe()


# In[6]:

list(hospital_read_df)


# In[7]:

#proportion of 'NA' or 'Not available' values in data
discharge_na = len(hospital_read_df[hospital_read_df['Number of Discharges'] == 'Not Available']) / len(hospital_read_df['Number of Discharges'])
excess_rr_na = len(hospital_read_df[hospital_read_df['Excess Readmission Ratio'].isnull()])/len(hospital_read_df['Excess Readmission Ratio'])
print(" The percentage of NA in discharge is", round(discharge_na*100,2),"%", "\n",  
      "The percentage of NA in excess readmission ratio is", round(excess_rr_na*100,2),"%", "\n")




# In[8]:

#proportion of na's in excess readmission ration which are also missing in number of discharges
err_na = hospital_read_df[hospital_read_df['Excess Readmission Ratio'].isnull()] 
err_nod_na = err_na[err_na['Number of Discharges'] == 'Not Available']
prop_na = len(err_nod_na)/len(err_na)
prop_na


# We will also remove the 3% missing NA's in Excess Readmission Ratio

# In[9]:

clean_hospital_read_df = hospital_read_df[hospital_read_df['Number of Discharges'] != 'Not Available']
clean_hospital_read_df = clean_hospital_read_df[clean_hospital_read_df['Excess Readmission Ratio'].notnull()] 
clean_hospital_read_df.loc[:, 'Number of Discharges'] = clean_hospital_read_df['Number of Discharges'].astype(int)
clean_hospital_read_df = clean_hospital_read_df.sort_values('Number of Discharges')
df = clean_hospital_read_df
df.head()


# A. Do you agree with the above analysis and recommendations? Why or why not?

# I do not agree with the analysis and recommendations yet. My point is that they do not provide statistical evidences supporting their hypothesis. Indeed, they just present the mean and percentage of excess readmission rate that is grater than 1.
# 

# B. Provide support for your arguments and your own recommendations with a statistically sound analysis:
# 

# Q 1. Setup an appropriate hypothesis test.

# According to the analysis, the recommendation is that smaller hospitals (with <100 discharges) have higher rates of readmission as compared to larger hospitals (with >1000 discharges). We will test this deduction statistically using CLT to test the following hypothesis

# H0: There is no difference between excess readmission rate in both groups.
# HA: There is a difference.

# First, let us check if the assumptions of CLT are satisfied.
# 
# Randomization Condition: The data must be sampled randomly.
# 
# Independence Assumption: The sample values must be independent of each other. 
# 
# Sample Size Assumption: The sample size must be sufficiently large.

# In[66]:

#Check the two samples


# In[18]:

#Sample for hospitals with discharges under 100
df100 = df[(df['Number of Discharges'] >0) & (df['Number of Discharges'] < 100)]
df100_mean = round(mean(df100["Excess Readmission Ratio"]),4)
df100_mean


# In[22]:

#df100_standanrd deviation
df100_sd = round((df100["Excess Readmission Ratio"]).std(),4)
df100_sd


# For the small hospitals sample, <100 discharges, the mean excess readmission ratio is 1.0226 and the standard deviation is 0.058

# In[24]:

#Sample for hospitals with discharges under 100
df1000 = df[df['Number of Discharges'] >1000]
df1000_mean = round(mean(df1000["Excess Readmission Ratio"]),4)
df1000_mean


# In[25]:

#df100_standanrd deviation
df1000_sd = round((df1000["Excess Readmission Ratio"]).std(),4)
df1000_sd


# For the small hospitals sample, <100 discharges, the mean excess readmission ratio is 0.9783 and the standard deviation is 0.12

# In[29]:

#Percentage of hospitals in under 100 with excess readmission rate more than 1
ratio100 = round(len(df100[(df100["Excess Readmission Ratio"]>1)])/len(df100)*100,2)
ratio100


# In[30]:

#Percentage of hospitals in over 1000 with excess readmission rate more than 1
ratio1000 = round(len(df1000[(df1000["Excess Readmission Ratio"]>1)])/len(df1000)*100,2)
ratio1000


# Plot the two samples to check normality, mean and sd

# In[32]:

#Plot for Hospitals with discharges under 100
df100["Excess Readmission Ratio"].plot(kind='hist',color='0.5', bins = 25, title = 'Histogram for # of admissions < 100').set_xlabel('Readmission rate')

print("The mean is {} and its standard deviation is {}".format(df100_mean, df100_sd))


# In[34]:

#Plot for Hospitals with discharges over 1000
df1000["Excess Readmission Ratio"].plot(kind='hist',color='0.5', bins = 25, title = 'Histogram for # of admissions > 1000').set_xlabel('Readmission rate')

print("The mean is {} and its standard deviation is {}".format(df1000_mean, df1000_sd))


# In[35]:

#Check sample size
len(df100)


# In[36]:

len(df1000)


# Since both samples have more than 100 observations, we can say that the samples are sufficiently large and the sampling distributions of the sample means is normally distributed

# Since each of these conditions is met, we can use the Z statistic to test the Hypothesis
# $H_0: \mu_s=\mu_l $ VS $H_0: \mu_s != \mu_l$ 
# Under the Null Hypothesis, we can define a test statistic $$Z^* = \frac{\bar X  - \bar Y }{\sqrt{(s1/n1)^2 + (s2/n2)^2 }}$$ 
# 
# which follows the Normal(0,1) distribution. If $H_0$ is true then we find the probability of getting $\bar X $ - $\bar Y $. If this probabilty i.e p-value is less than $\alpha$ = 0.05 then we reject the null hypothesis.

# Q 2. Compute and report the observed significance value (or p-value).

# In[53]:

#Let us first calculate difference in means, SE and Margin of error at alpha = 0.05
n1 = len(df100['Excess Readmission Ratio'])
n2 = len(df1000['Excess Readmission Ratio'])
diff_means = df100_mean - df1000_mean
se_diff = math.sqrt((df100_sd**2)/n1 + (df1000_sd**2)/n2)
zvalue = diff_means/se_diff
zvalue


# Now, we know that for a two tailed SN Distribution, the Z Critical for rejection at alpha = 0.05 is 1.96, since Z is 

# In[57]:

#Calculate the pvalue
p = round((1-stats.norm.cdf(zvalue))*2,5)
p


# Since the pvalue is less than 0.05, we reject the null hypothesis and conclude that there is a significant difference in Excess Readmission Ratio between both the groups

# Q3. Report statistical significance for  αα  = .01.

# In[56]:

#LCalculate p at alpha = 0.01
p = round((1-stats.norm.cdf(zvalue))*2,5)
p


# Since p value is less than 0.01, we still reject the null hypothesis in favor of the alternate hypothesis

# Let us calculate the effect size
# 
# $$Effect Size = \frac{\mu_x - \mu_y}{S.D. Pooled}$$
# 

# In[63]:

sd_pooled = math.sqrt(((n1-1)*((df100_sd)**2) + ((n2 -1)*((df1000_sd)**2)))/(n1+n2-2))
sd_pooled


# In[68]:

effect_size = round(diff_means/sd_pooled,2)
effect_size


# The Excess Readmission Ratio Score of an average hospital in the under 100 group is 0.55 standard deviations above the average hospital in the over 1000 group and hence exceeds 49% of the over 1000 group

# Q.4 Discuss statistical significance and practical significance. Do they differ here? How does this change your recommendation to the client?

# Here I provided strong arguments that support the fact that there is a statistical difference between readmision rate for hospitals with a number of admissions lower than 100 and this number for those with more than 1000. The high Z-score, the effect size and an almost null p-value support that hypothesis. This is why I conclude that the difference is a statistically significant.

# Q5 Look at the scatterplot above.
# What are the advantages and disadvantages of using this plot to convey information?
# Construct another plot that conveys the same information in a more direct manner.

# Scatter plot has the advantage of showing every single observation, which shows us the general picture, also we can clearly see the variance of readmission, and there might be some correlation between the variance of readmission and discharge, but the data is relatively sparse for big hospitals. The disadvantage of scatter plot is that when there are lots of data, they tend to concentrate, which reduces the visibility of data points. For the scatter plot above, the purpose of highlighting the areas in red and green is unclear. We can use bar plot to convey the same message.

# I agree with the recommendations that Hospitals/facilties with small capacity (< 300) should be required to demonstrate upgraded resource allocation for quality care to continue operation and directives and incentives should be provided for consolidation of hospitals and facilities to have a smaller number of them with higher capacity and number of discharges.

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



