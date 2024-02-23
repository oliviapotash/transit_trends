import numpy as np
import pandas as pd
import scipy.stats as stats

from statsmodels.miscmodels.ordinal_model import OrderedModel

url = "https://stats.idre.ucla.edu/stat/data/ologit.dta"
data_student = pd.read_stata(url)

print(data_student.head(5))

print(data_student.dtypes)

print(data_student['apply'].dtype.ordered)

mod_log = OrderedModel(data_student['apply'],
                        data_student[['pared']],
                        distr='logit')

res_log = mod_log.fit(method='bfgs', disp=False)
print(res_log.summary())