
# coding: utf-8

# # Examining Racial Discrimination in the US Job Market
# 
# ### Background
# Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.
# 
# ### Data
# In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.
# 
# Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

# <div class="span5 alert alert-info">
# ### Exercises
# You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.
# 
# Answer the following questions **in this notebook below and submit to your Github account**. 
# 
#    1. What test is appropriate for this problem? Does CLT apply?
#    2. What are the null and alternate hypotheses?
#    3. Compute margin of error, confidence interval, and p-value.
#    4. Write a story describing the statistical significance in the context or the original problem.
#    5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?
# 
# You can include written notes in notebook cells using Markdown: 
#    - In the control panel at the top, choose Cell > Cell Type > Markdown
#    - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# 
# 
# #### Resources
# + Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
# + Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# </div>
# ****

# In[32]:

import pandas as pd
import numpy as np
from scipy import stats
import math
import scipy.stats as stats


# In[5]:

data = pd.io.stata.read_stata('/Users/ruhama.ahale/Documents/Springboard_Coursework/racial_disc/us_job_market_discrimination.dta')


# In[6]:

# number of callbacks for black-sounding names
sum(data[data.race=='b'].call)


# In[7]:

data.head()


# In[8]:

data.describe()


# Q1. What test is appropriate for this problem? Does CLT apply?

# We should use the two sample z test to test whether the proportion of resumes selected were same for both the samples of black and white candidates. For CLT to apply, the following assumptions need to be true:
# 
# 1. Observations should be independent
# 2. Sample should be random, if sampling without replacement sample size n <10% of population
# 3. Sample should be sufficiently large
# 
#  a. There should be at least 10 successes and 10 failures in the sample
#  
#  b. np>=10 and n(1-p)>=10
# 

# Sample size is >1000 which is sufficiently large, and each resume is independent from other so observations are independent and since we know that sample is taken randomly we can use CLT. As the first two conditions are met, lets focus on the third. Since we dont have information on population proportion, let us use pooled proportion.

# $\hat{p}$Pool = $\frac{\ Total successes}{\ total  n} $ = $\frac{\ Number successes 1 + Number successes 2}{\ n1+n2} $ = $\frac{\ Number calls w + Number calls b}{\ n1+n2} $

# In[51]:

#Get seperate datasets for b and w:
df_white = data[data.race == 'w']
df_black = data[data.race == 'b']

#Get len of both data sets:
w_len = len(df_white.race)
b_len = len(df_black.race)

#get number of calls
w_calls = sum(df_white.call)
b_calls = sum(df_black.call)

#sample proportion
w_s_prop = w_calls/w_len
b_s_prop = b_calls/b_len

#diff in sample prop
s_p_diff = w_s_prop - b_s_prop

print(round(w_s_prop,3), round(b_s_prop,3))


# In[25]:

ppool= round((w_calls+b_calls)/(w_len+b_len),3)
ppool


# In[29]:

# Then let's check if n1*p_pool >=10, n1*(1-p_pool) >=10; (n1 = w_len, n2 = b_len)
np_w=w_len*ppool
n1p_w=w_len*(1-ppool)

# and for n2p_pool >=10, n2(1-p_pool) >=10:
np_b=b_len*ppool
n1p_b=b_len*(1-ppool)

print(np_w,n1p_w, np_b,n1p_b)


# Since all are >10, we can use CLT

# Q2. What are the null and alternate hypotheses?

# We want to test if a person that has a black name has different probability of getting a call back from a person that has a white name, such that we write our hypothesis testing $H_0: p_{black} - p_{white} = 0$ vs $H_1: p_{black} - p_{white} \neq 0$, which can also be written as $H_0: p_w = p_b$ vs $H_1: p_w \neq p_b $

# Q3. Compute margin of error, confidence interval, and p-value.

# Under the assumption that the null hypothesis is true, we have that
# 
# $ z = \frac{\hat{p}_{w} - \hat{p}_{b}}{\sqrt{{p(1-p)(\frac{1}{n_w} + \frac{1}{n_b}) }}}$ 
# 
# atleast approximately follows the Normal(0,1) distribution. Since we don't know the (assumed) common population proportion p any more than we know the proportions $p_w$ and $p_b$ of each population, we can estimate p using the pooled proportion earlier calculated i.e. ppool

# In[50]:

#Let us calculate the Standard Error for difference
se_w = math.sqrt((w_s_prop*(1-w_s_prop))/w_len)
se_b = math.sqrt((b_s_prop*(1-b_s_prop))/b_len)
se_diff = math.sqrt((se_w**2)+(se_b**2))
se_diff


# In[56]:

#Can also be calculated using pooled proportion using below:
#se_diff2 = math.sqrt(ppool*(1-ppool)*((1/w_len)+(1/b_len)))
#se_diff2


# Thus, a 95% Confidence Interval for the differences between these two proportions in the population is given by:
# 
# Difference Between the Sample Proportions ± z∗(Standard Error for Difference)

# In[55]:

#Calculating the zvalue
Zvalue= round((w_s_prop - b_s_prop)/se_diff,2)
Zvalue


# In[59]:

#calculating margin of error
moe = round(Zvalue*se_diff,3)
moe


# In[88]:

#Calculate confidence interval
#Difference Between the Sample Proportions±z∗(Standard Error for Difference)
#s_p_diff +- Zvalue*se_diff
lower_limit = round(s_p_diff - moe,3)
upper_limit = round(s_p_diff + moe,3)
print(lower_limit, upper_limit)


# In[90]:

print("The 95% confidence interval is (",lower_limit, ",", upper_limit,")")


# In[100]:

#since this is a two tailed hypothesis, we have to find the p value for Z >4.12 i.e. probablity
#that z falls in the rejection region and this will be true for both tails so  *2
p = (1-stats.norm.cdf(z))*2
p


# We reject the null hypothesis H0 if Z ≥ 1.96 or if Z ≤ −1.96. We clearly reject H0, since 4.12 is much greater than 1.96. There is sufficient evidence at the 0.05 level to conclude that the two populations differ with respect to the fact that race has a significant impact on the rate of callbacks for resumes.

# Q.4. Write a story describing the statistical significance in the context or the original problem.

# The proportion of callbacks for resumes are different with regards to the race of the applicant, this sample shows that there is significant difference in callbacks given to black vs white applicants

# Q5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

# No, this analysis only shows that race is a significant factor, not that it is the most important factor. There are several other attributes whose significance can be studied to the outcome of callbacks. Accordingly, I will amend my analysis saying race can be significant factor for callbacks, however there is need to study the importance of other applicant attributes as well. This can be done by fitting a regression model to the data.

# 

# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



