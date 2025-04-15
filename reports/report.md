# EDA Analysis Report

![Fraud Distribution Plot](fraud_distribution.png)

The visualization above suggests that the dataset to be quite imbalanced, with majority of the claims being Non-fraud.

![Fraud vs Non-Fraud Claims Reported by age_range Plot](age_range.png)

The plot suggests that people in the age range from 31-45 has a higher count of frauds as compared to other age range. This may be possibly due to financial stressors in younger policyholders.

![Fraud vs Non-Fraud Claims Reported by months_as_customer Plot](months_as_customer.png)

We can see that fraud claims are more prevalant when months_as_customer < 300, possibly due to opportunistic behaviour.

![Fraud vs Non-Fraud Claims Reported by insured_hobbies Plot](insured_hobbies.png)

An interesting insight is people that has chess as insured hobbies and cross-fit has significantly higher number of fraud reported compared to the rest.

![Fraud vs Non-Fraud Claims Reported by insured_occupation Plot](insured_occupation.png)

People in exec-managerial has more frauds committed than other occupations.

![Statewise Incident Count Plot](statewise_incident_count.png)

We can see that the state with the highest incident count is NY (New York) followed by SC (South Carolina).

![Statewise Fraud Count Plot](statewise_fraud_count.png)

From this choropleth map, we can see that SC has the highest fraud count instead followed by NY even though SC has slightly lower incident count compared to NY.

![Fraud vs Non-Fraud Claims Reported by insured_education_level Plot](insured_education_level.png)

From the chart above, we can see that the number of fraud/non-fraud claims are quite well distributed among the different insured education level.

![property_claim vs property_damage Plot](property_claim_vs_property_damage.png)

Fraud cases are more common when property damage is reported as 'YES' 78/302 = approx 25.8% as compared to when property damage is reported as 'NO' 169/698 = approx 19.5%. However, amount of fraud claims when property damage is reported as 'YES' is still more than when property damage is reported as 'NO'.

![property_claim_against_fraud_reported Plot](property_claim_against_fraud_reported.png)

The boxplot above shows that both fraud and non-fraud cases have clusters at lower claim values, with some high-value outliers, therefore following a similar distribution

![incident_type_vs_total_claim_amount Plot](incident_type_vs_total_claim_amount.png)

We can see that for each incident_type, fraud claims and non-fraud claims follows a similar claim distribution. However, there seems to be a larger percentage of fraud claims happening when the incident involves vehicle collision.

![incident_claim_against_fraud_reported Plot](incident_claim_against_fraud_reported.png)

Fraud cases have a slightly higher median injury claim, suggesting that fraudulent claims often involve larger compensation requests.

![total_claim_amount_vs_fraud_reported Plot](total_claim_amount_vs_fraud_reported.png)

We can see that the median for fraud claims is slightly higher compared to non-fraud claims, with outliers being both extremely high and extremely low claim amounts. This indicates that fraudsters may either inflate damages significantly (resulting in high-value claims) or fabricate small damages (resulting in low-value claims to avoid detection).

![bodily_injuries Plot](bodily_injuries.png)

We can see that fraud claims are quite well distributed despite the number of bodily injuries with bodily injuries = 2 being slightly higher.

![authorities_contacted Plot](authorities_contacted.png)

The number of fraud claims remains consistent, regardless of the authorities contacted. However, there appears to be a higher percentage of fraud claims when emergency services like ambulance or fire fighters are involved. This could suggest that incidents requiring such authorities tend to be of a larger scale and may present more opportunities for fraudulent activity.

![auto_year Plot](auto_year.png)

The number of fraud claims varies year over year.

![auto_make Plot](auto_make.png)

Car brands like Audi, Mercedes and Ford have a higher fraud rate as compared to the other brands.

![incident_severity Plot](incident_severity.png)

Claims for minor damage incidents are typically more common across all cities compared to those for other levels of incident severity.

![incident_severity_fraud Plot](incident_severity_fraud.png)

However, it is interesting how incident severity with Major Damage has a very large percentage of fraud claims as compared to the other incident severity.

![incident_hour_of_day Plot](incident_hour_of_day.png)

Fraud claims is inconsistent throughout the day.

![incident_month_year Plot](incident_month_year.png)

No meaningful insights can be drawn from the incident_month_year column, as it contains data for only three months. Furthermore, number of claims in 2015-01 and 2015-02 are significantly higher than 2015-03 suggesting that collection of data probably stopped early 2015-03.

![fraud_rate Plot](fraud_rate.png)

The fraud rate fluctuates significantly across different policy_bind periods. This suggests that fraud occurrences are inconsistent over time, potentially influenced by external factors as well.
