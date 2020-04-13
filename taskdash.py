# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:45:53 2020

@author: robbf
"""

'''
#----------------------------
Start by importing all dependencies
'''

import flask # The application dependency

# Bokeh is the graph modelling library used to display data
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d) # These tools help setup the bokeh plot
from bokeh.models.glyphs import VBar # VBar is the vertical bar display
from bokeh.plotting import figure # Figure plot, same as matplotlib
from bokeh.embed import components # Turns plots into html components
from bokeh.models import (ColumnDataSource, CDSView, GroupFilter, 
                          BooleanFilter, Panel, CustomJS) # Modelling data resources
from bokeh.models.widgets import Tabs, Panel, Div, Dropdown # Addons for displaying data
from bokeh.models.sources import AjaxDataSource # Pulling data from ajax
from bokeh.layouts import WidgetBox, column, row # Helping layout bokeh plots
from bokeh.core import json_encoder  # Encode to json

# From flask, import utilities for adding to the application
from flask import (
        Blueprint, flash, g, redirect, render_template, request, 
        session, url_for
        )

# Werkzeug security 
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug import secure_filename

import pandas as pd

from G1DA.db import get_db
from G1DA.auth import login_required

import os

import threading
import requests
import json

import PyPDF2 as pdf

import json

from zipfile import ZipFile

from . import ml_predict as mlp

from . import auth

from flask import Flask

import io

bp = Blueprint('taskdash',__name__)

server = flask.Flask(__name__)

@bp.route('/dash',methods=['GET','POST'])
@auth.login_required
def dashboard():
    task_all_df = load_source()
    data_task = task_all_df.groupby('IWA_actual_title')
    data_count = data_task.count()
    data_count['Position Time Spent'] = data_count['Position_title'].apply(lambda x: x/sum(data_count['Position_title']))
    data_count.reset_index(inplace=True)
    
    data_uniq = data_task.nunique()
    data_uniq.reset_index(drop=True,inplace=True)
    
    data_mean = data_task.mean()
    data_mean.reset_index(drop=True,inplace=True)
    
    data_task = pd.merge(data_count,data_uniq,left_index=True,right_index=True)
    data_task = pd.merge(data_task,data_mean,left_index=True,right_index=True)
    
    print(data_task.columns)
    
    data_task.sort_values('Position_title_x',ascending=False,inplace=True)
    
    data_task.drop(data_task[data_task['IWA_probability'] <= 0.1].index,inplace=True)

    plot = create_chart(data_task,"Task Overview", "IWA_actual_title_x", "Position_title_x", width=1200, height=800)

    #------------------
    
    data_title = task_all_df.groupby(['Position_title','IWA_actual_title'])
    data_title = data_title.count()
    
    data_title.sort_values('task',ascending=False,inplace=True)
    
    position_source, position_title, all_position_titles = make_position_dataset(data_title)
    
    position_selection = Dropdown(label="Select a position",menu=all_position_titles)
    
    callback = CustomJS(args=dict(source=position_source),code="""
                        function postData(input) {
                        $.ajax({
                        type: "POST";
                        url: "./_get_position/",
                        data: {'position_title':input,
                        'source':source},
                        success: callbackFunc
                        });
                        }
                        
                        function callbackFunc {
                        location.reload(forceGet=True);
                        
                        console.log("Refreshed")
                        }
                        
                        var f = cb_obj.value;
                        
                        postData(f);
                        
                        source.change.emit();
                        """)
                        
    position_selection.js_on_change('value',callback)
    
    tab1 = Panel(child=plot,title="Overview")
    tab2 = Panel(child=position_selection,title="Positions")
    
    tabs = Tabs(tabs=[tab1,tab2])
    
    script, div = components(tabs)
    
    positions = task_all_df['Position_title'].nunique()
    tasks = len(task_all_df['task'].unique())
    task_clusters = len(task_all_df['IWA_actual_id'].unique())
    
    return render_template('task/dashboard.html',
                           the_script=script,
                           the_div=div,
                           position_titles=all_position_titles,
                           positions=positions,
                           tasks=tasks,
                           clusters=task_clusters,
                           current_position_title=position_title)
    
def load_source():
    db = get_db()
    task_all_df = pd.read_sql("select * from task_overview",db)
    task_all_df.reset_index(inplace=True)
    return task_all_df

def create_chart(data,title,x_label,y_label,width,height):
    source = ColumnDataSource(data)
    
    plot = figure(x_range=data[x_label],title=title,plot_width=width,
                  plot_height=height)
    
    plot.vbar(x=x_label,top=y_label,width=0.9,source=source)
    
    plot.xaxis.major_label_orientation = 3.14/2
    
    hover = HoverTool(tooltips = [('Task cluster','@IWA_actual_title_x'),
                                  ('Clustered Tasks','@Position_title_x'),
                                  ('Positions','@Position_title_y'),
                                  ('Probability','@IWA_probability')])
    
    plot.add_tools(hover)
    
    return plot

def create_position_chart(source,title,width,height):
    
    position_view = CDSView(source=source,
                            filters=[GroupFilter(column_name='Position_title',
                                                 group=title)])
    
    x_range_list = list(source.data['IWA_actual_title'])
    
    plot = figure(x_range=FactorRange(*x_range_list),title=title,plot_width=width,plot_height=height)
    
    plot.vbar(x='IWA_actual_title',
              top='task',
              width=0.5,
              source=source,
              view=position_view)
    
    hover = HoverTool(tooltips = [('Position Title',title),
                                  ('Task','@IWA_actual_title'),
                                  ('Clustered Tasks','@task')])
    
    plot.add_tools(hover)
    
    plot.xaxis.major_label_orientation = 3.14/2
    
    return plot

def make_position_dataset(data,title=False,return_dataframe=False):
    
    if data.index.nlevels == 2:
        data = data.reset_index(level=[0,1])

    all_position_titles = [title for title in data['Position_title'].unique()]
    
    if title == False:
        title = all_position_titles[0]

    title_df = data[(data['Position_title'] == title) & (data['task'] >=1)]
    
    if return_dataframe:
        return title_df, title, all_position_titles
    else:
        return ColumnDataSource(title_df), title, all_position_titles

position_title = ''

@bp.route('/_get_position/',methods=['GET','POST'])
def get_position():
    if request.method == 'POST':
        task_all_df = load_source()
        
        data_title = task_all_df.groupby(['Position_title','IWA_actual_title'])
        data_title = data_title.count()
        
        data_title.sort_values('task',ascending=False,inplace=True)
        
        global position_title
        
        try:
            postData = request.form
            title = str(postData['position_title'].value)
        except:
            title = False
        
        position_source, position_title, all_position_titles = make_position_dataset(data_title,title)
        
        position_plot = create_position_chart(position_source,position_title,width=1200,height=800)   
        
        script, div = components(position_plot)
        
        return render_template('task/position.html',position_script=script,position_div=div)


def make_ajax_position_plot(json=None):
    
    task_all_df = load_source()
    data_title = task_all_df.groupby(['Position_title','IWA_actual_title'])
    data_title = data_title.count()
    data_title.sort_values('task',ascending=False,inplace=True)
    data = data_title.reset_index([0,1])
    
    global position_title
    
    try: 
        request.get_json((request.url_root + '_get_position/'))
    except:
        get_position()
    
    df_url = pd.read_json(request.url_root + '_get_position/')
    
    position_source, position_title, all_position_titles = make_position_dataset(df_url,position_title)
    
    position_plot = create_position_chart(position_source,position_title,width=1200,height=800)   
    
    script, div = components(position_plot)
    
    return render_template('task/position.html',position_script=script,position_div=div)
    
    
    
    