# Predicting Startup Success with Machine Learning

## Abstract

Venture Capital is a form of private equity investing that is known to be extremely high risk, with most companies having a high failure rate but with the potential for an extremely high return rate if things go well. In this project, we seek to create a model that can predict the success of a startup to better assist venture capital firms in making investment decisions. 

We obtained a dataset that contains information about startup companies and investments via Crunchbase, widely regarded as one of the premier sources of startup data. The data is composed of 39 columns and 49,439 observations. The columns consist of company names and founding dates, location information, industry information, and detailed funding information (covering all their rounds of venture capital funding). Startups in the dataset were founded between 1902 and 2014, which covers many notable events including the 2008 financial crisis, among others. 

We predicted the status of a startup through a 10-fold cross-validated logistic regression model regularized using Lasso with a train-test split of 4:1. We also used a 5-fold cross-validated Random Forest classification model, searching over hyperparameters including number of trees, minimum observations per split and leaf, and maximum depth, to predict the success feature of startups.  

Through data analysis, we found that startups were steadily gaining efficiency – the success rate of startups from 2007 to 2014 rose by 10% whereas the average total funding had decreased by 92%. For predicting the success of startups utilizing all relevant features, the random forest model had the best prediction accuracy with an AUC score of 0.76 against the logistic regression model’s AUC score of 0.68. 

By 2027, the venture capital industry is expected to grow to $580B USD and our findings illustrate how machine learning can be leveraged by venture capital firms to maximize the return on investment of their funds and elect to invest in startups that are most likely to succeed.

## Report

### Introduction  

Venture Capital is a form of private equity investing that is known to be extremely high risk, with most companies having a high failure rate but also an extremely high return rate if things go well. The general rule of thumb is that out of 10 startups invested, three to four will fail completely, three to four will recoup the initial investment, and one or two will produce a large return. While advanced analytics and models are beginning to emerge as a key influence in making VC investments, these decisions are still primarily made through investors’ past experiences. However, as exhibited through Ashenfelter’s wine prediction model, expertise can only consider a limited number of factors and is often biased. Considering that the current market size for venture capital is around $234 billion [1], a more advanced methodology for evaluating startups is desired and has broad societal implications for this industry.  

By combining machine learning concepts with startup data and consulting with a seasoned venture capitalist in the field, we were seeking to create a model that could predict the success of a startup, which we defined specifically as being “acquired or operating,” while addressing three key questions: Which features are most important for a startup to be successful? How accurately can we predict the success of a startup? Which model results in a better classification accuracy between random forest and logistic regression? 

### Data  

We used a Kaggle dataset (https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase) that contains information about startup companies and investments via Crunchbase. Crunchbase is widely regarded as one of the premier sources of startup data and therefore, its accuracy is reliable. Crunchbase gets its data from 4 key sources: a large investor network comprised of 4,000 global investment firms; active community contributors comprised of executives, entrepreneurs, and investors; AI and Machine Learning algorithms that validate data accuracy; and in house data analysts that validate and curate data. The data is composed of 39 columns and 49,439 observations. The columns consist of company names and founding dates, location information, industry information, and detailed funding information (covering all their rounds of venture capital funding and outside investments). See Appendix A for more detail. Startups in the dataset were founded between 1902 and 2014, which covers many notable events including the 2008 financial crisis, among others. The dataset we chose contains many null values for both numerical and categorical columns which was handled through data cleaning methods such as creating new features, and removing rows or columns, when necessary, as part of our EDA. 

 

### Exploratory Data Analysis  

Our initial dataset consisted of 14 features associated with funding data across various stages of funding. However, this also meant that there were a lot of missing or null values across all 14 features for a given startup. To treat this imbalance, we created new features (“pre-seed,” “early-stage VC,” and “late-stage”) based on a standard timeline of startup funding (Figure 1), which we obtained from a venture capitalist in the field. This also allowed us to convert our ultimate analysis into something that makes more sense for an investor to use, as these are the common subsets of the various stages of funding that startups can receive. Pre-seed typically refers to startups that are still in the ideation/development stage and typically get much smaller investments. Early-Stage VC is for early-stage companies whose business model has been successful and there is opportunity for growth. Late-stage is for companies with a proven business model with revenue to show for it and typically involves a large amount of money being invested. Subsequently, we grouped 13 out of those 14 features into the new features. The only feature that did not fit into this timeline was “debt-financing” since within a standard timeline of funding for a startup, it could receive debt-based financing at any stage that does not necessarily fall into one of the 3 categories. 

| ![Figure1.png](Images/Figure1.png?raw=true) | 
|:--:| 
| **Figure 1:** Standard timeline of startup funding, obtained from a venture capitalist |

We decided to use the “status” column as our target in whether a startup is successful or not. This feature takes the class values of “operating,” “acquired,” and “closed,” which we have split up into success (comprised of the two former) and failure (comprised of the latter). Operating companies comprised 86.21% of the data (27,183 observations), with acquired and closed making up 5.40% and 8.39%, respectively. This means that our success class was ~91.6%, with failure being ~8.4%.  

Visualizing total investment by funding stage, we saw that late-stage funding comprised the largest share, with very minimal investment comprised of pre-seed funding sources, which made sense given the explanation of the categories above.  


| ![Figure2.png](Images/Figure2.png?raw=true) | 
|:--:| 
| **Figure 2:** Total investment by funding stage, showing that late-stage funding comprises the largest share |


Diving into late-stage funding (comprised of VC funding rounds B-H), we observed that the three main funding rounds from late-stage VC funding are Rounds B, C, and D, with approximately $60B, $51B, and $32B in funding, respectively (Figure 2). The remaining four rounds comprised less than 25% of the total investment, which we attributed to the fact that many of the successful firms will go public or begin operating from self-sustained revenue after only a few rounds of VC funding. 

The heatmap of correlated features shown in Appendix B1 (with corresponding R^2 values) allowed us to immediately notice a few highly correlated features. Some of the correlated features of interest were Rounds A-H and venture, which are due to venture capital being the primary source of rounds A-H funding. Round G was highly correlated with Round H funding, and total investment had some degree of expected correlation with many of the other monetary features, as it is a summation of the monetary features. 

Analyzing the total investment and number of companies by the market type (shown in Appendix B2), we saw that although there are over 4000 companies in the Software market, the highest amount of funding was received by companies in the Biotechnology market. A similar trend followed for other markets as well, as there was not an exact, direct, and proportional relationship between the number of companies in a market and the total investment by market, which is an important insight when making investment decisions.  

Through splitting the homepage URL feature, we were able to isolate the URL-ending (i.e. .com, .net) from each observation’s URL. As shown in Figure 3, the most common URL-ending, .com, comprised over 90% of total companies, so we visualized these plots excluding .com URL endings. We saw that .net is the most prominent outside of .com with over 600 startups. We also saw that .co is the second highest in number of startups but only ninth highest in total investments (see Appendix B3). This, combined with differences in success rates (see Appendix B4), indicated that having a more common website ending may not indicate more total investment or success rate, which we investigated further in our model by including URL-ending as a feature. Additionally, we computed the age of the startup at the first date it received funding, and included it as a feature, by computing the difference between the date the startup was founded at and the date it received its first funding.  
 
 
| ![Figure3.png](Images/Figure3.png?raw=true) | 
|:--:| 
| **Figure 3:** Number of companies by URL ending (left), showing that .net is the most prominent URL ending outside of .com with over 600 startups and $5B of investment, and number of companies by name length (right), showing a slightly right-skewed distribution centered around 8-9 characters |
 

Next, we wanted to investigate the relationship between the average success rate and the founded year of the startup. In Figure 4, we can see that the average success rate steadily declined from 1995 to 2007 and steadily increased thereafter. We more specifically observed that during the Great Recession of 2008, the average success rate of startups was 4% less than the average of the dataset. However, post-recession, from 2010-2011 (inclusive of both) and 2012-2013 (inclusive of both), the average success rate increased by 4% in comparison to the previous timeline.  


| ![Figure4.png](Images/Figure4.png?raw=true) | 
|:--:| 
| **Figure 4:** Success Rate vs Timelines and Founded Year, showing the increase in success rates post-recession |


Since the year the startup was founded in ranged from 1904 to 2014, all startups founded before 1995 were removed. This constituted 3.92% of the data or 1228 observations (see Appendix, B5). To remove any additional small, confounding patterns in the data occurring before 1995 due to the lack of data for those years, we made the decision to only include data starting with the year 1995.  

### Methods 

#### One-Hot Encoding and Dimensionality Reduction 

To utilize categorical features such as ‘market’, ‘country-code’, ‘quarter_founded’, and ‘URL-ending’, one-hot encoding was used to create binary columns of the categorical features of interest. However, there were 716 distinct markets with skewed class distributions. For instance, 407 (57%) of markets consisted of less than 10 observations each, constituting only 5% of the total observations. Additionally, the top-10 markets constituted 43% of the observations. A similar argument can be made for the country-code and URL-ending features as well, with approximately 90% and 95% of the observations belonging to the 15-top occurring country-codes and 10-top occurring URL-endings, respectively. Therefore, to balance the size of classes and reduce the creation of additional columns, the dimensionality of the dataset was reduced by only considering the 20-top occurring markets, 15-top occurring country-codes, and 10-top occurring URL-endings. An “other” class, for each of the categorical variables, was also created by grouping the remaining observations. The result of adding this dummy class within each feature has been summarized in the table below (Table 1).  


| ![Table1.png](Images/Table1.png?raw=true) | 
|:--:| 
| **Table 1:** Table of market, country, and URL-ending features highlighting quantity of observations partitioned from minority classes into “other” class as a raw count of observations and percentage of total observations  |
 

#### Models 

For the creation of our machine learning models, we first partitioned our dataset into 80% training and 20% testing using a random seed of 1. We trained three logistic regression models using GridSearchCV in the sklearn package in Python [4]. We used 10 cross-validations, the ‘liblinear’ solver, and tuned hyperparameters such as ‘C’ over 8 values ranging from 0.001 to 1000 and ‘penalty’ over ‘l1’ (Lasso) and ‘l2’ (Ridge). All other hyperparameters were set to their default values. Additionally, all numerical features were transformed by scaling to [0,1] using MinMaxScaler in the sklearn package in Python. Appendix A visualizes the features used in the baseline model. 

Next, we trained two random forest models – one containing all features (as the baseline of our logistic regression model did) and one excluding the year a startup was founded. Various hyperparameters were optimized through the use of grid search including maximum depth of individual trees, minimum samples per decision tree split, and number of features considered per split. For hyperparameter selection, we performed 5-fold cross-validation using hyperparameter choices including ‘n_estimators’ at four values between 25 and 100, ‘max_depth’ at five values ranging from 5 to 25, ‘min_samples_split’ at four values from 50 to 400, ‘min_samples_leaf’ ranging from 3 to 30, and ‘max_features’ ranging from 2 to 12. 

### Results 

#### Logistic Regression 

The table of ROC AUC scores and feature selections (Table 2) and the ROC curves (Figure 5) are shown below. For each of these models, we set C equal to 0.1 (which is the inverse of regularization strength as per sklearn documentation), max iterations to 10,000, and set the penalty to L1 (for L1 regularization).  

| ![Table2.png](Images/Table2.png?raw=true) | 
|:--:| 
| **Table 2:** Feature selection and ROC-AUC scores for three logistic regression models |


| ![Figure5.png](Images/Figure5.png?raw=true) | 
|:--:| 
| **Figure 5:** ROC curves for logistic regression models showing True Positive Rate and False Positive Rate tradeoff for different discrimination thresholds |


#### Random Forest 

The optimal hyperparameter combination to maximize the ROC curve AUC for the model containing all features consisted of 100 trees, 50 samples minimum per split, 5 samples minimum per leaf, a maximum depth of 25 nodes, and a maximum of 12 features considered per split for an AUC of 0.81. The optimal hyperparameter combination to maximize the AUC for the model excluding year founded consisted of 100 trees, 150 samples minimum per split, 5 samples minimum per leaf, a maximum depth of 15 nodes, and a maximum of 6 features considered per split resulting in an AUC of 0.67.  The table of ROC AUC scores and feature selections (Table 3) and the ROC curves (Figure 6) are shown below. 


| ![Table3.png](Images/Table3.png?raw=true) | 
|:--:| 
| **Table 3:** Feature selection and ROC-AUC scores for three random forest models |


| ![Figure6.png](Images/Figure6.png?raw=true) | 
|:--:| 
| **Figure 6:** ROC curves for random forest models showing True Positive Rate and False Positive Rate tradeoff for different discrimination thresholds |
 

| ![Figure7.png](Images/Figure7.png?raw=true) | 
|:--:| 
| **Figure 7:** Confusion matrices for logistic regression (left) and random forest (right) models showing intersection of actual target classes and target predictions |

Looking at the confusion matrices of the logistic regression and random forest models with the highest AUC scores, respectively, there is a clear difference between the two models in terms of correct and incorrect classifications (Figure 7). Both models had high precision, with 96.5% for logistic regression and 97.9% for random forest. Recall was much lower with 85.1% and 70.0%, respectively. Specificity was also lower with 50.1% and 74.6% for the two models, respectively. False positives (type I errors) made up 2.9% of the logistic regression classifications and 1.4% of the random forest classifications, while false negatives (type II errors) made up 14.1% of the logistic regression classifications and 28.3% of the random forest classifications. 


### Discussion 

#### Interpretability and Prediction 

Both the logistic regression models and random forest models we ran emphasized interpretability. With our logistic regression model, we can take coefficients generated from the model and interpret those coefficients as an increase or decrease in the log of odds of our status target. We can also interpret features that had their coefficients reduced to zero as being less important in explanation of the success of a startup. On the other hand, with our random forest models, we were able to derive feature importances using mean decrease in impurity across the individual CART models that were assembled in the ensemble (Figure 8). From this bar chart, we can see that the age a startup received its first funding, the year a startup was founded, the funding a startup receives in early stages, the length of the company name, and the amount of funding rounds a startup receives were the most important features in classifying the startups’ success.  


| ![Figure8.png](Images/Figure8.png?raw=true) | 
|:--:| 
| **Figure 8:** Bar chart of mean decrease in impurity (MDI) showing median (blue bar) and variance (black line) of MDI for features with median MDI greater than or equal to 0.005 |

Given our intent to use our models in a predictive manner, we saw it fit to create a logistic regression model without year as a feature. Since founded_year was an important feature in both models’ AUC values, patterns in the data tied to the year cannot necessarily be used to extrapolate patterns in the data if startups founded in more recent years (past the year 2014) were to be added to the dataset. We saw the AUC heavily affected by the removal of this feature from the logistic regression model, where it decreased from 0.79 to 0.67. 

#### Errors 

Classification errors were a major point of difference between the best models of logistic regression and random forest. The logistic regression and random forest models both had a precision above 96% but varied widely in terms of type I and type II errors. The logistic regression had approximately 700 less type II errors than the random forest model but had 86 more type I errors. In the context of the startup success problem, type II errors indicate a successful startup that is missed by the model, whereas type I errors denote a model predicting success (and in action terms, triggering an investment) for a company that ends up failing. With this context in mind, type I errors are much more severe, as these are situations where an investor is losing money on their investment. Type II errors are simply startups that the model fails to detect as successful, so in theory, no money is invested and therefore no money is lost. The money lost on the 86 additional type I errors outweighs the potential gain of the 700 extra type II errors, so we elected the random forest model to be the most successful in minimizing the risk of making bad business decisions. 

#### Limitations

This model has some key limitations that may affect how accurately it can predict a startup’s success. In terms of the data, our features did not include things like revenue, the background of the founding team, market potential, and more, which are key things that investors look at when making decisions. Our data is also from past years and technology, investment philosophies, and features that make a company successful may have changed in more recent years. Additionally, as mentioned previously, the models presented here are significantly better at predicting success than failure. While we do prefer to commit type II errors as opposed to type I errors, the potential investment returns are still being lost when committing a type II error, and our model commits a high rate of type II errors. Finally, our data is very imbalanced, with 94.5% of the data being a success, which could have implications for our results having the proportion of Type II errors that it does (especially since we are much better at predicting success than failure).  

### Conclusions and Future Directions  

By 2027, the venture capital industry is expected to grow to $584.4B USD [5] and our findings illustrate how machine learning can be leveraged by venture capital firms to maximize the return on investment of their funds and elect to invest in startups that are most likely to succeed. After speaking to a venture capitalist who works in the field and explaining our model, the features, and logic behind it, he agreed that our analysis is a great starting point for widespread usage of a model like this.  

Referencing the key limitations discussed in the Discussion section, there are numerous future directions that could be taken with a model like this. In terms of data collection, if we had additional features that venture capitalists look at when making investments such as revenue, the background of the founding team, and initial traction of the company’s idea, the model would be a lot more robust and accurate. To solve the problem of imbalanced data, we can explore more K-Fold cross validation methods such as stratified sampling, under sampling of the majority (success) class, and oversampling of the minority (failure) class. Finally, some of our data was time sensitive as well, so methods such as ARIMA could also be explored to further validate our model.  

 

### References  

[1] Venture Capital Investment Market: Global Industry Trends, share, size, growth, opportunity and forecast 2023-2028 (2022) Share and Size 2023-2028. Imarc Group. Available at: https://www.imarcgroup.com/venture-capital-investment-market  

[2] M, A. (2020) Startup investments (crunchbase), Kaggle. Kaggle. Available at: https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase 

[3] Team, C.P. (2022) Where does Crunchbase get their data?, crunchbase.com. Crunchbase. Available at: https://support.crunchbase.com/hc/en-us/articles/360009616013-Where-does-Crunchbase-get-their-data- 

[4] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. 

Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. 		Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011. 

 

[5] Anderson, E. (2022) Venture Capital Investment Market to reach US$ 584.4 billion by 2027, catalyzed by rising number of start-ups, EIN News. EIN Presswire. Available at: https://www.einnews.com/pr_news/593165703/venture-capital-investment-market-to-reach-us-584-4-billion-by-2027-catalyzed-by-rising-number-of-start-ups  

 

 

 

 

 

 

 

### Appendix A: Description of features and the key features used during modeling 

The columns consist of company names and founding dates, location information, industry information, and detailed funding information (covering all their rounds of venture capital funding and outside investments). 

 

Company names: This refers to the names of each company which we used to reference and confirm some numbers during the initial EDA.  

Founding dates: This refers to the month and year that the startup was founded.  

Location information: This refers to the region that the startup was based in (at different granularities of city and country).  

Industry information: This refers to each startup’s industry/market.  

Detailed funding information: This refers to all the different types of funding that each startup received, covering angel investors, venture funding, and all the different rounds of venture funding. Venture funding for early-stage companies typically begins with round A and continues to Round B, C, D, etc. as the company grows and adds more investors.  

 

The graph below lists all the features used for the baseline models of Logistic Regression and Random Forest, and how we transformed the features for use in the model.   

 

### Appendix B: Exploratory Data Analysis (EDA) and key insights from EDA 

EDA key insights  

Late-stage funding comprised the largest share, with very minimal investment comprised of pre-seed funding sources, which made sense given the explanation of the categories above. 

Diving into late-stage funding (comprised of VC funding rounds B-H), we observed that the three main funding rounds from late-stage VC funding are Rounds B, C, and D, with approximately $60B, $51B, and $32B in funding, respectively. The remaining four rounds comprised less than 25% of the total investment, which we attributed to the fact that many of the “success” firms will go public or begin operating from self-sustained revenue after only a few rounds of VC funding. 

Some of the correlated features of interest were Rounds A-H and venture, which are likely due to venture capital being the primary source of rounds A-H funding. Round G was highly correlated with Round H funding, and total investment had some degree of expected correlation with many of the other monetary features as it is a summation of the monetary features. 

There are over 4000 companies in the Software market, the highest amount of funding was received by companies in the Biotechnology market. A similar trend followed for other markets as well, as there was not an exact direct proportional relationship between the number of companies and the total investment by market, which is an important insight when making investment decisions. 

The most common URL-ending, .com, comprised over 85% of the total investment and over 90% of total companies, so we visualized these plots excluding .com URL-endings. We saw that .net is the most prominent outside of .com with over 600 startups and $5B of investment. We also saw that .co is the second highest in # of startups but only ninth highest in total investments. This, combined with other differences, indicated that having a more common website ending may not indicate more total investment 

 

EDA supplemental graphs  

Heatmap of correlated features, showing that Rounds A-H and venture, and Round G and Round H are correlated. 

 
XXX
 

Total investment by market type, showing that the Biotechnology market received the highest amount of funding. 

 
XXX
 

Total investment received by startups grouped by URL-ending. 

 
XXX
 

The success rate of startups by URL-ending. 

 
XXX
 


Startups founded by year. The startups founded before 1995 form a small subset of the dataset. 

 
