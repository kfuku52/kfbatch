import pandas

import re
import subprocess
import sys

def get_qstat_df(lines):
    lines = [ re.sub('\n$', '', l) for l in lines ]
    lines = [ l for l in lines if l!='' ]
    lines = [ l for l in lines if not l.startswith('queuename') ]
    lines = [ l for l in lines if not l.startswith('---') ]
    lines = [ l for l in lines if not l.startswith('###') ]
    lines = [ l for l in lines if not l.startswith(' ') ]
    lines = [ l for l in lines if not l.startswith('\n') ]
    node_params = {}
    df = pandas.DataFrame()
    for line in lines:
        if not line.startswith('\t'):
            if len(node_params.keys())>0:
                tmp = pandas.DataFrame(node_params, index=[0,])
                df = pandas.concat([df,tmp], axis=0, ignore_index=True)
            node_params = {}
            items = [ item for item in line.split(' ') if item!='' ]
            node_params['queue_name'] = re.sub('@.*', '', items[0])
            node_params['node_name'] = re.sub('.*@', '', items[0])
            node_params['qtype'] = items[1]
            node_params['ncore_resv'] = re.sub(r'([0-9]+)/([0-9]+)/([0-9]+)', r'\1', items[2])
            node_params['ncore_used'] = re.sub(r'([0-9]+)/([0-9]+)/([0-9]+)', r'\2', items[2])
            node_params['ncore_total'] = re.sub(r'([0-9]+)/([0-9]+)/([0-9]+)', r'\3', items[2])
            node_params['np_load'] = items[3]
            node_params['arch'] = items[4]
            if len(items)>5:
                node_params['status'] = items[5]
            else:
                node_params['status'] = ''
        else:
            key = re.sub('\t', '', line)
            key = re.sub('=.*', '', key)
            value = re.sub('.*=', '', line)
            node_params[key] = value
    for col in ['ncore_resv','ncore_used','ncore_total']:
        df[col] = df[col].astype(int)
    df.loc[:,'ncore_available'] = df['ncore_total'] - df['ncore_used'] - df['ncore_resv']
    df = df.sort_values(by=['queue_name','node_name']).reset_index(drop=True)
    return df

def print_stats(df):
    for i in df.index:
        queue_name = df.at[i, 'queue_name']
        num_avail_cpu = df.at[i, 'ncore_available']
        avail_ram = df.at[i, 'hc:mem_req']
        ram_unit = df.at[i, 'hc:mem_req_unit']
        node_name = df.at[i, 'node_name']
        node_status = df.at[i, 'status']
        txt = '{}: {:,} cores and {:,.0f}{} RAM in {}'
        if node_status!='':
            txt += ' with the status {}'
        print(txt.format(queue_name, num_avail_cpu, avail_ram, ram_unit, node_name, node_status))

def print_resource_availability(df, args):
    queue_names = df.loc[:,'queue_name'].unique()
    queue_names = [ q for q in queue_names if not q.startswith('login') ]
    resources = dict()
    resources['RAM'] = 'hc:mem_req'
    resources['core'] = 'ncore_available'
    for resource_name in resources.keys():
        col = resources[resource_name]
        print('Reporting top {} availability:'.format(resource_name))
        for queue_name in queue_names:
            if args.exclude_abnormal_node:
                df_queue = df.loc[(df['queue_name']==queue_name)&(df['status']==''),:]
            else:
                df_queue = df.loc[(df['queue_name']==queue_name),:]
            other_cols = [ oc for oc in list(resources.values()) if oc!=col ]
            sort_by = [col, ] + other_cols
            df_queue = df_queue.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
            if args.all_tiers:
                descending_values = df_queue[col]
                threshold_value = descending_values.iloc[min(args.ntop-1, descending_values.shape[0]-1)]
                df_top_availability = df_queue.loc[(df_queue[col]>=threshold_value),:]
                df_top_availability = df_top_availability.sort_values(by=col, ascending=False).reset_index(drop=True)
            else:
                df_top_availability = df_queue.iloc[0:args.ntop,:]
            print_stats(df=df_top_availability)
        print('')

def get_user_df(lines):
    ser = pandas.Series(lines)
    is_user_line = ser.str.match(r'^  [0-9]* ')
    ser = ser[is_user_line]
    ser = ser.str.replace(r'\n$', '', regex=True)
    ser = ser.str.replace(r'^  ', '', regex=True)
    df_user = pandas.DataFrame(ser.str.split(' +', regex=True).tolist())
    df_user.columns = ['job_id','prior','name','user','state','submit_or_start_date','submit_or_start_time','slots','ja_task_id']
    df_user['slots'] = df_user['slots'].astype(int)
    df_user['total_slots'] = 0
    for i in df_user.index:
        task_id = df_user.at[i,'ja_task_id']
        if ',' in task_id:
            df_user.at[i,'total_slots'] = df_user.at[i,'slots'] * 2
        elif '-' in task_id:
            start = re.sub(r'([0-9]+)-([0-9]+):([0-9]+)', r'\1', task_id)
            end = re.sub(r'([0-9]+)-([0-9]+):([0-9]+)', r'\2', task_id)
            step = re.sub(r'([0-9]+)-([0-9]+):([0-9]+)', r'\3', task_id)
            total_slots = int(df_user.at[i,'slots'] * (int(end) - int(start) + 1) / int(step))
            df_user.at[i, 'total_slots'] = total_slots
        else:
            df_user.at[i, 'total_slots'] = df_user.at[i, 'slots']
    return df_user

def print_queued_job_summary(df_user):
    is_running = df_user['state'].str.contains('r', regex=False)
    is_halted = df_user['state'].str.contains('h', regex=False)
    is_qwaiting = df_user['state'].str.contains('qw', regex=False)
    is_error = df_user['state'].str.contains('E', regex=False)
    num_running = df_user.loc[is_running,'total_slots'].sum()
    num_halted = df_user.loc[is_halted,'total_slots'].sum()
    num_qwaiting = df_user.loc[is_qwaiting,'total_slots'].sum()
    num_error = df_user.loc[is_error,'total_slots'].sum()
    print('# of CPUs in use for running jobs: {}'.format(num_running))
    print('# of requested CPUs for queued jobs: {}'.format(num_qwaiting))
    print('# of CPUs for queued/running jobs in error: {}'.format(num_error))
    print('')

def get_df(args):
    for i in range(args.niter):
        if (args.example_file != ''):
            with open(args.example_file) as f:
                lines = f.readlines()
        else:
            command = args.stat_command.split(' ')
            command_out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            command_stdout = command_out.stdout.decode('utf8')
            lines = command_stdout.split('\n')
        if (args.stat_command == 'qstat -F'):
            df_i = get_qstat_df(lines)
        else:
            print('Exiting. --stat_command does not support: {}'.format(args.stat_command))
            exit(1)
        if i==0:
            df = df_i
            df_user = get_user_df(lines)
            print_queued_job_summary(df_user)
        else:
            for col in ['ncore_available','hc:mem_req']:
                is_less_available = df[col]>df_i[col]
                df.loc[is_less_available,col] = df_i.loc[is_less_available,col]
    return df

def adjust_ram_unit(df):
    for col in ['hc:mem_req','hl:mem_total']:
        df[col+'_unit'] = df[col].str.replace(r'^[\.0-9]+([A-Z]*)$', r'\1', regex=True).fillna('')
        df[col] = df[col].str.replace('[A-Z]$', '', regex=True).astype(float)
        is_t = (df[col+'_unit']=='T').fillna(False)
        if is_t.sum():
            df.loc[is_t,col] = df.loc[is_t,col] * 1000
            df.loc[is_t, col+'_unit'] = 'G'
        is_m = (df[col+'_unit']=='M').fillna(False)
        if is_m.sum():
            df.loc[is_m,col] = df.loc[is_t,col] * 0.001
            df.loc[is_m, col+'_unit'] = 'G'
    return df
def print_cluster_summary(df):
    queue_names = df['queue_name'].unique()
    print('Reporting working/abnormal/total nodes, available/used/reserved/abnormal/total CPUs, and available/total RAM:')
    for queue_name in queue_names:
        df_queue = df.loc[(df['queue_name']==queue_name),:].reset_index(drop=True)
        is_abnormal_status = (df_queue['status']!='')
        num_abnormal_node = is_abnormal_status.sum()
        num_node = df_queue.shape[0]
        num_working_node = num_node - num_abnormal_node
        ncore_total = df_queue.loc[:,'ncore_total'].sum()
        ncore_used = df_queue.loc[~is_abnormal_status,'ncore_used'].sum()
        ncore_reserved = df_queue.loc[~is_abnormal_status,'ncore_resv'].sum()
        ncore_abnormal = df_queue.loc[is_abnormal_status,'ncore_total'].sum()
        ncore_available = df_queue.loc[~is_abnormal_status,'ncore_available'].sum()
        mem_total = df_queue.loc[:,'hl:mem_total'].sum()
        mem_available = df_queue.loc[~is_abnormal_status,'hc:mem_req'].sum()
        txt = '{}: {}/{}/{} nodes, {}/{}/{}/{}/{} CPUs, and {:,.0f}/{:,.0f}G RAM'
        print(txt.format(queue_name,
                         num_working_node, num_abnormal_node, num_node,
                         ncore_available, ncore_used, ncore_reserved, ncore_abnormal, ncore_total,
                         mem_available, mem_total))
    print('')

def stat_main(args):
    df = get_df(args)
    df = adjust_ram_unit(df)
    print_cluster_summary(df)
    print_resource_availability(df, args)
    if args.out!='':
        df.to_csv(args.out, sep='\t', index=False)
