'''
This module contains a number of custom functions to facilitate data analysis. 
It is mostly based on code by Mauro Di Pietro (https://github.com/mdipietro09)
that I have revisited and adapted for my own needs.
Umberto Selva (https://github.com/umbertoselva)
'''

# IMPORTS  -------------------------------------------------------------------------------

## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for testing
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# for machine learning
from sklearn import metrics



# CUSTOM FUNCTIONS  -----------------------------------------------------------------------


# FOR EDA ---------------------------------------------------------------------------------


def recognize_col_type(df, col, max_cat=20):

    '''
    Recognize whether a column is numerical or categorical.
    :parameters
        :param df: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of uniques to consider a variable as categorical
    :return
        "cat" if the column is categorical, 
        "dt" if datetime, 
        "num" otherwise
    '''

    if (df[col].dtype == "O") | (df[col].nunique() < max_cat):
        return "cat" # Categorical
    elif df[col].dtype in ['datetime64[ns]','<M8[ns]']:
        return "dt" # Datetime
    else:
        return "num" # Numerical


def df_overview(df, max_cat=20, figsize=(10,5)):

    '''
    Get a general overview of a dataframe.
    :parameters
        :param df: dataframe - input data
        :param max_cat: num - max number of uniques to consider a variable as categorical
    '''

    ## recognize column type
    col_dict = {col : recognize_col_type(df, col, max_cat=max_cat) for col in df.columns}
        
    ## print info
    len_df = len(df)
    print("Shape:", df.shape)
    print("-----------------")
    for col in df.columns:
        info = col + " --> Type:" + col_dict[col]
        info = info + " | NaN values: " + str(df[col].isna().sum()) + "(" + str(int(df[col].isna().mean()*100)) + "%)"
        if col_dict[col] == "cat":
            info = info + " | Categories: " + str(df[col].nunique())
        elif col_dict[col] == "dt":
            info = info + " | Range: " + "({x})-({y})".format(x=str(df[col].min()), y=str(df[col].max()))
        else:
            info = info + " | Min-Max: " + "({x})-({y})".format(x=str(int(df[col].min())), y=str(int(df[col].max())))
        if df[col].nunique() == len_df:
            info = info + " | Possible Primary Key"
        print(info)
                
    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = df.isnull()
    for key, value in col_dict.items():
        if value == "num":
            heatmap[key] = heatmap[key].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[key] = heatmap[key].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, cmap='crest', ax=ax).set_title('Dataset Overview')
    # plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[48;2;125;186;145m Light Green = Categorical \033[m", 
          "\033[48;2;64;144;142m Dark Green = Numerical \033[m",
          "\033[48;2;37;75;127m Blue = NaN \033[m")


def univariate_plots(df, x, max_cat=20, top=None, show_perc=True, bins=100, quantile_breaks=(0,10), box_logscale=False, figsize=(10,5)):
    
    '''
    Plots a univariate analysis, the frequency distribution of a df column.
    :parameters
        :param df: dataframe - input data
        :param x: str - column name
        :param max_cat: num - max number of uniques to consider a variable as categorical
        :param top: num - plot setting
        :param show_perc: logic - plot setting
        :param bins: num - plot setting
        :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
        :param box_logscale: logic
        :param figsize: tuple - plot settings
    '''

    try:
        
        ## cat --> freq (barplot)
        if recognize_col_type(df, x, max_cat) == "cat":   
            ax = df[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
            ax.grid(axis="x")
            plt.suptitle(x, fontsize=20)
            plt.show()
            
        ## num --> density plot & boxplot
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            
            ### distribution
            ax[0].title.set_text('distribution')
            variable = df[x].fillna(df[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade":True}, ax=ax[0])
            des = df[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            
            ### boxplot 
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_df = pd.DataFrame(df[x])
                tmp_df[x] = np.log(tmp_df[x])
                tmp_df.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                df.boxplot(column=x, ax=ax[1])
            plt.show()   
        
    except Exception as e:
        print("--- got error ---")
        print(e)


def bivariate_plots(df, x, y, max_cat=20, figsize=(10,5)):

    '''
    Plots a bivariate analysis.
    :parameters
        :param df: dataframe - input data
        :param x: str - column
        :param y: str - column
        :param max_cat: num - max number of uniques to consider a variable as categorical
    '''
    
    try:
       
        ## num vs num --> stacked + scatter with density
        if (recognize_col_type(df, x, max_cat) == "num") & (recognize_col_type(df, y, max_cat) == "num"):
            ### stacked
            df_noNan = df[df[x].notnull()]  #can't have nan
            breaks = np.quantile(df_noNan[x], q=np.linspace(0, 1, 11))
            groups = df_noNan.groupby([pd.cut(df_noNan[x], bins=breaks, duplicates='drop')])[y].agg(['mean','median','size'])
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            groups[["mean", "median"]].plot(kind="line", ax=ax)
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True)
            ax.set(ylabel=y)
            ax.right_ax.set_ylabel("Observazions in each bin")
            plt.show()
            ### joint plot
            sns.jointplot(x=x, y=y, data=df, dropna=True, kind='reg', height=int((figsize[0]+figsize[1])/2) )
            plt.show()

        ## cat vs cat --> hist count + hist %
        elif (recognize_col_type(df, x, max_cat) == "cat") & (recognize_col_type(df, y, max_cat) == "cat"):  
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### count
            ax[0].title.set_text('count')
            order = df.groupby(x)[y].count().index.tolist()
            sns.catplot(x=x, hue=y, data=df, kind='count', order=order, ax=ax[0])
            ax[0].grid(True)
            ### percentage
            ax[1].title.set_text('percentage')
            a = df.groupby(x)[y].count().reset_index()
            a = a.rename(columns={y:"tot"})
            b = df.groupby([x,y])[y].count()
            b = b.rename(columns={y:0}).reset_index()
            b = b.merge(a, how="left")
            b["%"] = b[0] / b["tot"] *100
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1]).get_legend().remove()
            ax[1].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()
    
        ## num vs cat --> density + stacked + boxplot 
        else:
            if (recognize_col_type(df, x, max_cat) == "cat"):
                cat,num = x,y
            else:
                cat,num = y,x
            
            # 2 cols only
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)

            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(df[cat].unique()):
                sns.distplot(df[df[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)

            ### stacked
            df_noNan = df[df[num].notnull()]  #can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(df_noNan[num], q=np.linspace(0,1,11))
            tmp = df_noNan.groupby([cat, pd.cut(df_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[df_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(df[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)

            ### fix figure
            plt.close(2)
            plt.close(3)

            sns.catplot(x=cat, y=num, data=df, kind="box", order=sorted(df[cat].unique()))
            plt.title('box')

            plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)



# FOR TESTING -----------------------------------------------------------------------------


def correlation_tests(df, x, y, max_cat=20, summary=False):

    '''
    Run correlation tests
        num vs num -> Pearson R, OLS Regression
        cat vs cat -> Chi-squared
        cat vs num -> Anova LM
    :parameters
        :param df: dataframe - input data
        :param x: str - feature column - independent variable
        :param y: str - target column - dependent variable
        :param max_cat: num - max number of uniques to consider a variable as categorical
        :param summary: bool - whether to print the test summary table
    '''

    try:
        # num vs num [indep continuous vs dep continuous]

        # suggestion: make scatterplot first

        # test
        if (recognize_col_type(df, x, max_cat) == "num") & (recognize_col_type(df, y, max_cat) == "num"):
            
            df_noNaN = df[df[x].notnull()]  # remove NaN values

            # Pearson R
            pearson_coeff, p = scipy.stats.pearsonr(df_noNaN[x], df_noNaN[y])
            pearson_coeff, p_value = round(pearson_coeff, 3), round(p, 3)
            conclusions_PR = "Significant (p-value < 0.05)" if p_value < 0.05 else "Non-Significant (p-value > 0.05)"
            print("Pearson R test result:", conclusions_PR)
            print("P-value:", p, "or", p_value)
            print("Pearson coefficient:", pearson_coeff)
            if pearson_coeff <= -0.5:
                print("Strong negative correlation")
            elif pearson_coeff > -0.5 and pearson_coeff < 0:
                print("Slight negative correlation")
            elif pearson_coeff >= 0 and pearson_coeff < 0.5:
                print("Slight positive correlation")
            elif pearson_coeff >= -0.5:
                print("Strong positive correlation")
            print()

            # OLS Regression
            # sm.OLS(y, X)
            X = df_noNaN[x]
            y = df_noNaN[y]
            X = sm.add_constant(X)      
            model = sm.OLS(y, X).fit()
            predictions = model.predict(X)
            p_value_reg = model.pvalues[1]
            conclusions_OLS = "Significant (p-value < 0.05)" if p_value_reg < 0.05 else "Non-Significant (p-value > 0.05)"
            print('OLS Regression test results:', conclusions_OLS)
            print('P-value:', p_value_reg, 'or', round(p_value_reg, 3))
            print('R-squared:', round(model.rsquared, 3))
            print('Regression coefficient:', round(model.params[x], 3))
            sign = np.sign(model.params[x])
            pc_abs = np.sqrt(model.rsquared)
            pc = pc_abs if sign == 1 or sign == 0 else np.negative(pc_abs)
            print("Pearson coefficient:", round(pc, 3))
            if pc <= -0.5:
                print("Strong negative correlation")
            elif pc > -0.5 and pc < 0:
                print("Slight negative correlation")
            elif pc >= 0 and pc < 0.5:
                print("Slight positive correlation")
            elif pc >= -0.5:
                print("Strong positive correlation")   
            if summary==True:
                print()
                print(model.summary())
                print()   

        # cat vs cat
        elif (recognize_col_type(df, x, max_cat) == "cat") & (recognize_col_type(df, y, max_cat) == "cat"):

            # cross table
            cont_table = pd.crosstab(index=df[x], columns=df[y])
            print(cont_table)
            print()

            # chi-squared test
            chi_sqr_results = scipy.stats.chi2_contingency(cont_table, correction=False)
            chi_sqr_value, p_value_chi, dof = chi_sqr_results[0], chi_sqr_results[1], chi_sqr_results[2]

            # Cramer's V
            n_obsv = cont_table.sum().sum() # total num of observations, sample size
            mindim = min(cont_table.shape)-1 # minimum dimension - 1
            v = np.sqrt(chi_sqr_value / (n_obsv * mindim))

            conclusions_chi = "Significant (p-value < 0.05)" if p_value_chi < 0.05 else "Non-Significant (p-value > 0.05)"
            print("Chi-squared Test results:", conclusions_chi)
            print("Chi-squared:", chi_sqr_value)
            print('P-value:', p_value_chi, 'or', round(p_value_chi, 3))
            print('Degrees of freedom:', dof, 'of', n_obsv)
            print('Cramer\'s V:', round(v, 3))
            if v < 0.1:
                print("Little if any association")
            elif v >= 0.1 and v < 0.3:
                print("Weak association")
            elif v >= 0.3 and v < 0.5:
                print("Moderate association")
            elif v >= 0.5:
                print("High association") 

        # cat vs num [independent categ vs dep continuous]
        elif (recognize_col_type(df, x, max_cat) == "cat") & (recognize_col_type(df, y, max_cat) == "num"):

            # Linear Model Anova

            # NB: ols('y ~ x', df)
            lm = ols(y+' ~ '+x, data=df).fit()
            table = sm.stats.anova_lm(lm)

            if summary==True:
                print(table)
                print()  

            f_statistic = table["F"][0]
            p_value_lm = table["PR(>F)"][0]
            conclusions_lm = "Significant (p-value < 0.05)" if p_value_lm < 0.05 else "Non-Significant (p-value > 0.05)"
            conclusions_lm2 = "correlated" if p_value_lm < 0.05 else "non-correlated"

            print("Anova LM test results:", conclusions_lm)
            print("F-statistic:", f_statistic)
            print("P-value:", p_value_lm, 'or', round(p_value_lm, 3))
            print("The variables are", conclusions_lm2)

        
    except Exception as e:
        print("--- got error ---")
        print(e)



# FOR PREPROCESSING ------------------------------------------------------------------------


def encode_cat(df, cat, drop_orig=True, drop_first=True, dummy_na=False):

    '''
    One-hot encoding for non-numerical categorical columns
    E.g. From an original column AC (values Y vs N) 
    two cols will be created AC_Y and AC_Y with only 0 vs 1 values 
    (corresponding to the Y and N of the original col)
    :parameters
        :param df: df[cat] - the non-numerical cat column
        :param prefix: str - name of the column to encode
        :param drop_orig: bool - if True, remove the original cat col
        :param drop_first: bool - if True, one of the two created col is removed
        :param dummy_na: bool - add a column to indicate NaNs, if False NaNs are ignored.
    :return
        df - the df containing the encoded col(s)
    '''

    # create dummy df
    df_dummy = pd.get_dummies(df[cat], prefix=cat, drop_first=drop_first, dummy_na=dummy_na)

    # concatenate features + dummy features + Y; make sure Y is at the end
    df = pd.concat([df.drop('Y', axis=1), df_dummy, df['Y']], axis=1) 

    # drop original feature
    if drop_orig == True:
        df = df.drop(cat, axis=1)

    return df



# FOR MODEL DESIGN ----------------------------------------------------------------------


def plot_keras_loss(training):

    '''
    Plot loss (in black) and metrics (in blue) of keras training.
    :inputs
        :training: an obj returned by keras.model.fit()
    '''

    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()


def evaluate_regr_model(y_test, predicted, figsize=(25,5)):

    '''
    Evaluates a model performance.
    :parameters
        :param y_test: array
        :param predicted: array
        :param figsize: tuple (e.g. (25, 5))
    '''

    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("MAPE - Mean Absolute Perc Error (Σ(|y-pred|/y)/n):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
    print("MAE - Mean Absolute Error (Σ|y-pred|/n):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("RMSE - Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    
    ## residuals (errors)
    residuals = y_test - predicted
    max_error = max(residuals) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(residuals).index(min(residuals))
    max_true, max_pred = y_test[max_idx], predicted[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))
    
    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].vlines(x=max_pred, ymin=max_true, ymax=max_true-max_error, color='red', linestyle='--', alpha=0.7, label="max error")
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    ax[0].legend()
    
    ## Plot predicted vs residuals
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].vlines(x=max_pred, ymin=0, ymax=max_error, color='black', linestyle='--', alpha=0.7, label="max error")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Predicted vs Residuals")
    ax[1].hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    ax[1].legend()
    
    ## Plot residuals distribution
    sns.distplot(residuals, color="red", hist=True, kde=True, kde_kws={"shade":True}, ax=ax[2], label="mean = "+"{:,.0f}".format(np.mean(residuals)))
    ax[2].grid(True)
    ax[2].set(yticks=[], yticklabels=[], title="Residuals distribution")
    plt.show()