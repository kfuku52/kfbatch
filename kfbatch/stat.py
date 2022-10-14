import pandas

import re
import subprocess

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
    df = df.sort_values(by='queue_name').reset_index(drop=True)
    for col in ['ncore_resv','ncore_used','ncore_total']:
        df.loc[:,col] = df.loc[:,col].astype(int)
    df.loc[:,'ncore_available'] = df['ncore_total'] - df['ncore_used'] - df['ncore_resv']
    return df

def print_stats(df):
    for i in df.index:
        queue_name = df.at[i, 'queue_name']
        num_avail_cpu = df.at[i, 'ncore_available']
        avail_ram = df.at[i, 'hc:mem_req']
        ram_unit = df.at[i, 'hc:mem_req_unit']
        node_name = df.at[i, 'node_name']
        node_status = df.at[i, 'status']
        txt = '{}: {:,} cores and {:.2f}{} RAM in {}'
        if node_status!='':
            txt += ' with the status {}'
        print(txt.format(queue_name, num_avail_cpu, avail_ram, ram_unit, node_name, node_status))

def print_resource_availability(df):
    df.loc[:,'hc:mem_req_unit'] = df.loc[:,'hc:mem_req'].str.replace(r'^[\.0-9]+([A-Z]*)$', r'\1', regex=True).fillna('')
    df.loc[:,'hc:mem_req'] = df.loc[:,'hc:mem_req'].str.replace('[A-Z]$', '', regex=True).astype(float)
    queue_names = df.loc[:,'queue_name'].unique()
    queue_names = [ q for q in queue_names if not q.startswith('login') ]
    for resource_name, col in zip(['RAM', 'core'], ['hc:mem_req', 'ncore_available']):
        print('Reporting top {} availability:'.format(resource_name))
        for queue_name in queue_names:
            df_queue = df.loc[(df['queue_name']==queue_name),:]
            descending_values = df_queue[col].sort_values(ascending=False)
            third_value = descending_values.iloc[min(2, descending_values.shape[0]-1)]
            df_top_available_ram = df_queue.loc[(df_queue[col]>=third_value),:]
            df_top_available_ram = df_top_available_ram.sort_values(by=col, ascending=False).reset_index(drop=True)
            print_stats(df=df_top_available_ram)
        print('')

def stat_main(args):
    if (args.example_file != ''):
        with open(args.example_file) as f:
            lines = f.readlines()
    else:
        command = args.stat_command.split(' ')
        command_out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command_stdout = command_out.stdout.decode('utf8')
        lines = command_stdout.split('\n')
    if (args.stat_command == 'qstat -F'):
        df = get_qstat_df(lines)
        print_resource_availability(df)
    else:
        print('Exiting. --stat_command does not support: {}'.format(args.stat_command))
        exit(1)
    if args.out!='':
        df.to_csv(args.out, sep='\t', index=False)
