# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:30:43 2020

@author: robbf
"""

from flask import (
        Blueprint, flash, g, redirect, render_template, request, session, url_for
        )

from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug import secure_filename

import pandas as pd

from G1DA.db import get_db
from G1DA.auth import login_required

import os

import PyPDF2 as pdf

from zipfile import ZipFile

from . import ml_predict as mlp

from . import auth

from . import taskdash

from flask import Flask

bp = Blueprint('taskview',__name__)

app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir,exist_ok=True)


@bp.route('/', methods=('GET','POST'))
@auth.login_required
def index():
    if request.method == 'POST':
        g.table = True
        tasks = []
        titles = []
        
        for file in request.files.getlist('filetoupload'):
            title = secure_filename(file.filename)
            title = title[:-4]
            title = title.split("_")
            title = " ".join(title)
            for raw_task, raw_title in mlp.pdf_to_text(file,title=title):
                tasks.append(raw_task)
                titles.append(raw_title)
                
        pd.set_option('display.max_colwidth', -1)
        dataframe = mlp.run_predictions(tasks,titles)
        
        dataframe.sort_values('IWA_probability',inplace=True,ascending=False)
        
        dataframe
        
        grouped = dataframe.groupby('IWA_actual_title')
        dataframe_count = grouped.count()
        dataframe_count['Position Time Spent'] = dataframe_count['Position_title'].apply(lambda x: x/sum(dataframe_count['Position_title']))
        
        dataframe_sum = grouped.mean()
        
        dataframe_uniq = grouped.nunique()
        
        dataframe_master = pd.merge(dataframe_count,dataframe_sum,left_index=True,right_index=True)
        dataframe_master = pd.merge(dataframe_master,dataframe_uniq,left_index=True,right_index=True)
        
        #dataframe_master.drop(['task_x','IWA_clusterid_x','IWA_actual_id_x','IWA_clusterid_y','IWA_probability_x','task_y','IWA_clusterid','IWA_actual_id_y','IWA_probability_y','IWA_actual_title'],
                              #axis=1,
                              #inplace=True)
        
        #dataframe_master.columns = ['% of All Tasks','Positions Performing this Task']
        
        db = get_db()
        
        dataframe_master.sort_values('Position Time Spent',ascending=False,inplace=True)
        
        dataframe_master.to_sql('task_db',db,if_exists='replace',index=False)
        dataframe.to_sql('task_overview',db,if_exists='replace',index=False)    
        
        dataframe.drop(['IWA_clusterid','GWA_clusterid'],axis=1,inplace=True)
        master_columns = list(dataframe_master.columns)
        master_columns.remove('Position Time Spent')
        master_columns.remove('Position_title_y')
        dataframe_master.drop(master_columns,axis=1,inplace=True)
        dataframe_master.reset_index(inplace=True)
        
        return redirect(url_for('taskdash.dashboard'))
    
    g.table = False
    return render_template('task/upload.html')

