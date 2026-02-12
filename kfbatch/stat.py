import pandas

import os
import re
import shlex
import subprocess

SLURM_RUNNING_STATES = {'R', 'CG'}
SLURM_PENDING_STATES = {'PD', 'CF'}
SLURM_ERROR_STATES = {
    'BF',  # BOOT_FAIL
    'CA',  # CANCELLED
    'DL',  # DEADLINE
    'F',   # FAILED
    'NF',  # NODE_FAIL
    'OOM', # OUT_OF_MEMORY
    'PR',  # PREEMPTED
    'RV',  # REVOKED
    'SE',  # SPECIAL_EXIT
    'ST',  # STOPPED
    'TO',  # TIMEOUT
}
SLURM_STATE_NAME_TO_CODE = {
    'RUNNING': 'R',
    'COMPLETING': 'CG',
    'PENDING': 'PD',
    'CONFIGURING': 'CF',
    'BOOT_FAIL': 'BF',
    'CANCELLED': 'CA',
    'DEADLINE': 'DL',
    'FAILED': 'F',
    'NODE_FAIL': 'NF',
    'OUT_OF_MEMORY': 'OOM',
    'PREEMPTED': 'PR',
    'REVOKED': 'RV',
    'SPECIAL_EXIT': 'SE',
    'STOPPED': 'ST',
    'TIMEOUT': 'TO',
}
SLURM_NORMAL_NODE_STATES = {'IDLE', 'MIXED', 'ALLOCATED', 'COMPLETING'}
SLURM_UNAVAILABLE_NODE_FLAGS = {
    'DRAIN',
    'DRAINING',
    'DOWN',
    'FAIL',
    'NOT_RESPONDING',
    'MAINT',
    'POWER_DOWN',
    'POWERING_DOWN',
    'POWERED_DOWN',
    'REBOOT_REQUESTED',
    'REBOOT_ISSUED',
    'PLANNED',
    'RESERVED',
}
SLURM_SQUEUE_PARSE_FIELDS = '%i\t%P\t%j\t%u\t%t\t%M\t%D\t%R'
QSTAT_REQUIRED_NODE_FIELDS = {
    'queue_name',
    'node_name',
    'qtype',
    'ncore_resv',
    'ncore_used',
    'ncore_total',
    'np_load',
    'arch',
    'status',
}


def _count_uge_task_expression(task_expression):
    if task_expression=='':
        return 1
    num_tasks = 0
    for token in task_expression.split(','):
        token = token.strip()
        if token=='':
            continue
        m = re.match(r'^([0-9]+)-([0-9]+):([0-9]+)$', token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            step = int(m.group(3))
            if (step<=0) or (end<start):
                continue
            num_tasks += int((end - start) / step) + 1
            continue
        m = re.match(r'^([0-9]+)-([0-9]+)$', token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            if end<start:
                continue
            num_tasks += int(end - start) + 1
            continue
        if re.match(r'^[0-9]+$', token):
            num_tasks += 1
    if num_tasks==0:
        return 1
    return num_tasks

def get_qstat_df(lines):
    columns = [
        'queue_name',
        'node_name',
        'qtype',
        'ncore_resv',
        'ncore_used',
        'ncore_total',
        'np_load',
        'arch',
        'status',
        'hc:mem_req',
        'hl:mem_total',
        'ncore_available',
    ]
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
            if QSTAT_REQUIRED_NODE_FIELDS.issubset(set(node_params.keys())):
                tmp = pandas.DataFrame(node_params, index=[0,])
                df = pandas.concat([df,tmp], axis=0, ignore_index=True)
            node_params = {}
            items = [ item for item in line.split(' ') if item!='' ]
            if len(items)<5:
                continue
            m = re.match(r'^([0-9]+)/([0-9]+)/([0-9]+)$', items[2])
            if m is None:
                continue
            node_params['queue_name'] = re.sub('@.*', '', items[0])
            node_params['node_name'] = re.sub('.*@', '', items[0])
            node_params['qtype'] = items[1]
            node_params['ncore_resv'] = m.group(1)
            node_params['ncore_used'] = m.group(2)
            node_params['ncore_total'] = m.group(3)
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
    if QSTAT_REQUIRED_NODE_FIELDS.issubset(set(node_params.keys())):
        tmp = pandas.DataFrame(node_params, index=[0,])
        df = pandas.concat([df,tmp], axis=0, ignore_index=True)
    if df.shape[0]==0:
        return pandas.DataFrame(columns=columns)
    for col in ['ncore_resv','ncore_used','ncore_total']:
        df[col] = df[col].astype(int)
    for mem_col in ['hc:mem_req', 'hl:mem_total']:
        if mem_col not in df.columns:
            df[mem_col] = '0G'
        df[mem_col] = df[mem_col].fillna('0G').astype(str)
        is_empty = (df[mem_col].str.strip()=='')
        if is_empty.sum():
            df.loc[is_empty, mem_col] = '0G'
    ncore_available = df['ncore_total'] - df['ncore_used'] - df['ncore_resv']
    ncore_available = ncore_available.clip(lower=0)
    tmp = pandas.DataFrame({'ncore_available': ncore_available.astype(int)})
    df = pandas.concat([df, tmp], axis=1)
    df = df.sort_values(by=['queue_name','node_name']).reset_index(drop=True)
    return df

def _memory_series_to_gib(series):
    raw = series.fillna('').astype(str).str.strip()
    units = raw.str.extract(r'([A-Za-z]+)$', expand=False).fillna('').str.upper()
    numeric_txt = raw.str.replace(r'[A-Za-z]+$', '', regex=True)
    values = pandas.to_numeric(numeric_txt, errors='coerce').fillna(0.0).astype(float)
    is_t = units.str.startswith('T')
    if is_t.sum():
        values.loc[is_t] = values.loc[is_t] * 1000
    is_m = units.str.startswith('M')
    if is_m.sum():
        values.loc[is_m] = values.loc[is_m] * 0.001
    is_k = units.str.startswith('K')
    if is_k.sum():
        values.loc[is_k] = values.loc[is_k] * 0.000001
    return values

def _merge_qstat_iteration_min_availability(df, df_i):
    key_cols = ['queue_name', 'node_name']
    if df.shape[0]==0:
        return df_i.copy()
    if df_i.shape[0]==0:
        return df
    if (not set(key_cols).issubset(set(df.columns))) or (not set(key_cols).issubset(set(df_i.columns))):
        return df
    df_base = df.set_index(key_cols, drop=False).copy()
    df_new = df_i.set_index(key_cols, drop=False)
    common_index = df_base.index.intersection(df_new.index)
    if len(common_index)==0:
        return df
    base_cores = pandas.to_numeric(df_base.loc[common_index, 'ncore_available'], errors='coerce').fillna(0)
    new_cores = pandas.to_numeric(df_new.loc[common_index, 'ncore_available'], errors='coerce').fillna(0)
    min_cores = pandas.concat([base_cores, new_cores], axis=1).min(axis=1)
    df_base.loc[common_index, 'ncore_available'] = min_cores.astype(int)
    base_mem = _memory_series_to_gib(df_base.loc[common_index, 'hc:mem_req'])
    new_mem = _memory_series_to_gib(df_new.loc[common_index, 'hc:mem_req'])
    min_mem = pandas.concat([base_mem, new_mem], axis=1).min(axis=1)
    df_base.loc[common_index, 'hc:mem_req'] = min_mem.map(lambda x: '{:.3f}G'.format(float(x)))
    df_base = df_base.reset_index(drop=True)
    df_base = df_base.sort_values(by=key_cols).reset_index(drop=True)
    return df_base

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
            if df_queue.shape[0]==0:
                continue
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
    columns = ['job_id','prior','name','user','state','submit_or_start_date','submit_or_start_time','slots','ja_task_id']
    if ser.shape[0]==0:
        return pandas.DataFrame(columns=columns + ['total_slots'])
    df_user = pandas.DataFrame(ser.str.split(' +', regex=True).tolist())
    df_user.columns = columns
    df_user['slots'] = df_user['slots'].astype(int)
    df_user['total_slots'] = 0
    for i in df_user.index:
        task_id = df_user.at[i,'ja_task_id']
        num_tasks = _count_uge_task_expression(task_id)
        df_user.at[i, 'total_slots'] = df_user.at[i, 'slots'] * num_tasks
    return df_user

def _count_slurm_array_task_expression(task_expression):
    if task_expression=='':
        return 1, True
    num_tasks = 0
    has_ambiguous_pattern = False
    for token in task_expression.split(','):
        token = token.strip()
        if token=='':
            has_ambiguous_pattern = True
            continue
        m = re.match(r'^([0-9]+)-([0-9]+)(?::([0-9]+))?$', token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            step = 1 if m.group(3) is None else int(m.group(3))
            if (step<=0) or (end<start):
                has_ambiguous_pattern = True
                continue
            num_tasks += int((end - start) / step) + 1
            continue
        if re.match(r'^[0-9]+$', token):
            num_tasks += 1
            continue
        has_ambiguous_pattern = True
    if num_tasks==0:
        return 1, True
    return num_tasks, has_ambiguous_pattern

def estimate_slurm_task_count(job_id):
    if '_' not in job_id:
        return 1, False
    job_suffix = job_id.split('_', 1)[1]
    if re.match(r'^[0-9]+$', job_suffix):
        return 1, False
    if not job_suffix.startswith('['):
        return 1, True
    task_expression = re.sub(r'^\[', '', job_suffix)
    has_closing_bracket = (']' in task_expression)
    if has_closing_bracket:
        task_expression = task_expression.split(']', 1)[0]
    task_expression = task_expression.split('%', 1)[0]
    num_tasks, has_ambiguous_pattern = _count_slurm_array_task_expression(task_expression)
    is_estimated = has_ambiguous_pattern or (not has_closing_bracket)
    return num_tasks, is_estimated

def get_squeue_user_df(lines):
    columns = [
        'job_id',
        'partition',
        'name',
        'user',
        'state',
        'elapsed_time',
        'num_nodes',
        'node_or_reason',
        'total_slots',
        'task_count_estimated',
    ]
    table = []
    for raw_line in lines:
        line = re.sub('\n$', '', raw_line)
        if line.strip()=='':
            continue
        if line.lstrip().startswith('JOBID '):
            continue
        if '\t' in line:
            items = line.split('\t', 7)
            if len(items)<8:
                continue
            job_id = items[0].strip()
            partition = items[1].strip()
            name = items[2].strip()
            user = items[3].strip()
            state = items[4].strip()
            elapsed_time = items[5].strip()
            num_nodes_txt = items[6].strip()
            node_or_reason = items[7].strip()
        elif '\\t' in line:
            # Some captured files may contain literal "\t" separators.
            items = line.split('\\t', 7)
            if len(items)<8:
                continue
            job_id = items[0].strip()
            partition = items[1].strip()
            name = items[2].strip()
            user = items[3].strip()
            state = items[4].strip()
            elapsed_time = items[5].strip()
            num_nodes_txt = items[6].strip()
            node_or_reason = items[7].strip()
        else:
            items = re.split(r'\s+', line.strip(), maxsplit=7)
            if len(items)<8:
                continue
            job_id = items[0]
            partition = items[1]
            name = items[2]
            user = items[3]
            state = items[4]
            elapsed_time = items[5]
            num_nodes_txt = items[6]
            node_or_reason = items[7]
        try:
            num_nodes = int(num_nodes_txt)
        except ValueError:
            num_nodes = 1
        num_tasks, is_estimated = estimate_slurm_task_count(job_id)
        total_slots = num_tasks
        table.append({
            'job_id': job_id,
            'partition': partition,
            'name': name,
            'user': user,
            'state': state,
            'elapsed_time': elapsed_time,
            'num_nodes': num_nodes,
            'node_or_reason': node_or_reason,
            'total_slots': total_slots,
            'task_count_estimated': is_estimated,
        })
    return pandas.DataFrame(table, columns=columns)

def _split_scontrol_node_blocks(lines):
    blocks = []
    current = ''
    for raw_line in lines:
        line = raw_line.strip()
        if line=='':
            if current!='':
                blocks.append(current.strip())
                current = ''
            continue
        if ('NodeName=' in line) and (current!=''):
            blocks.append(current.strip())
            current = line
            continue
        if current=='':
            current = line
        else:
            current += ' ' + line
    if current!='':
        blocks.append(current.strip())
    return blocks

def _parse_key_value_fields(line):
    items = [ item for item in line.split(' ') if item!='' ]
    params = {}
    for item in items:
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        params[key] = value
    return params

def _safe_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def _partition_state_is_up(partition_state):
    state = str(partition_state).strip().upper()
    if state=='':
        return True
    tokens = re.findall(r'[A-Z_]+', state)
    if len(tokens)==0:
        return True
    return (tokens[0]=='UP') and (len(tokens)==1)

def _normalize_slurm_node_state(state_raw):
    if state_raw=='':
        return ''
    m = re.match(r'^([A-Z]+)', state_raw.upper())
    if m is None:
        return state_raw.upper()
    return m.group(1)

def _slurm_state_flags(state_raw):
    if state_raw=='':
        return []
    flags = []
    for token in state_raw.upper().split('+'):
        m = re.match(r'^([A-Z_]+)', token)
        if m is None:
            continue
        flags.append(m.group(1))
    return flags

def get_scontrol_partition_df(lines):
    columns = ['partition_name', 'partition_state']
    rows = []
    for raw_line in lines:
        line = raw_line.strip()
        if line=='':
            continue
        if 'PartitionName=' not in line:
            continue
        params = _parse_key_value_fields(line)
        partition_name = params.get('PartitionName', '')
        partition_state = params.get('State', '')
        if partition_name=='':
            continue
        rows.append({
            'partition_name': partition_name,
            'partition_state': partition_state,
        })
    return pandas.DataFrame(rows, columns=columns)

def get_scontrol_node_df(lines, partition_state_map=None):
    columns = [
        'queue_name',
        'node_name',
        'qtype',
        'ncore_resv',
        'ncore_used',
        'ncore_total',
        'ncore_available',
        'np_load',
        'arch',
        'status',
        'hl:mem_total',
        'hc:mem_req',
        'slurm_state',
    ]
    rows = []
    node_blocks = _split_scontrol_node_blocks(lines)
    for node_block in node_blocks:
        if 'NodeName=' not in node_block:
            continue
        params = _parse_key_value_fields(node_block)
        node_name = params.get('NodeName', '')
        if node_name=='':
            continue
        partition_raw = params.get('Partitions', '')
        partitions = [ p.strip().rstrip('*') for p in partition_raw.split(',') if p.strip()!='' ]
        partitions = [ p for p in partitions if p not in ['(null)', 'N/A'] ]
        if len(partitions)==0:
            continue
        ncore_total = _safe_int(params.get('CPUEfctv', ''), default=0)
        if ncore_total<=0:
            ncore_total = _safe_int(params.get('CPUTot', ''), default=0)
        ncore_total = max(ncore_total, 0)
        ncore_used = max(_safe_int(params.get('CPUAlloc', ''), default=0), 0)
        ncore_resv = 0
        ncore_available = max(ncore_total - ncore_used - ncore_resv, 0)
        mem_total_mb = max(_safe_int(params.get('RealMemory', ''), default=0), 0)
        mem_available_mb = _safe_int(params.get('FreeMem', ''), default=-1)
        if mem_available_mb<0:
            alloc_mem_mb = _safe_int(params.get('AllocMem', ''), default=-1)
            if alloc_mem_mb>=0:
                mem_available_mb = max(mem_total_mb - alloc_mem_mb, 0)
            else:
                mem_available_mb = 0
        else:
            mem_available_mb = max(mem_available_mb, 0)
        slurm_state = params.get('State', '')
        state_base = _normalize_slurm_node_state(slurm_state)
        flags = _slurm_state_flags(slurm_state)
        has_unavailable_flag = any((flag in SLURM_UNAVAILABLE_NODE_FLAGS) for flag in flags)
        node_status = '' if ((state_base in SLURM_NORMAL_NODE_STATES) and (not has_unavailable_flag)) else slurm_state
        arch = params.get('Arch', '')
        for partition in partitions:
            partition_status = ''
            if partition_state_map is not None:
                partition_state = partition_state_map.get(partition, '')
                if not _partition_state_is_up(partition_state):
                    partition_status = 'partition_state={}'.format(partition_state)
            status = node_status
            if (status!='') and (partition_status!=''):
                status = '{}|{}'.format(status, partition_status)
            elif partition_status!='':
                status = partition_status
            rows.append({
                'queue_name': partition,
                'node_name': node_name,
                'qtype': 'SLURM',
                'ncore_resv': ncore_resv,
                'ncore_used': ncore_used,
                'ncore_total': ncore_total,
                'ncore_available': ncore_available,
                'np_load': '',
                'arch': arch,
                'status': status,
                'hl:mem_total': '{}M'.format(mem_total_mb),
                'hc:mem_req': '{}M'.format(mem_available_mb),
                'slurm_state': slurm_state,
            })
    df = pandas.DataFrame(rows, columns=columns)
    if df.shape[0]==0:
        return df
    for col in ['ncore_resv', 'ncore_used', 'ncore_total', 'ncore_available']:
        df[col] = df[col].astype(int)
    df = df.sort_values(by=['queue_name', 'node_name']).reset_index(drop=True)
    return df

def _normalize_slurm_job_state(state_raw):
    if state_raw is None:
        return ''
    state = str(state_raw).strip().upper()
    if state=='':
        return ''
    m = re.match(r'^([A-Z_]+)', state)
    if m is not None:
        state = m.group(1)
    return SLURM_STATE_NAME_TO_CODE.get(state, state)

def print_queued_job_summary(df_user, scheduler='uge'):
    if scheduler=='slurm':
        if df_user.shape[0]==0:
            print('No jobs found in squeue output.')
            print('')
            return
        state_codes = df_user['state'].fillna('').map(_normalize_slurm_job_state)
        is_running = state_codes.isin(SLURM_RUNNING_STATES)
        is_qwaiting = state_codes.isin(SLURM_PENDING_STATES)
        is_error = state_codes.isin(SLURM_ERROR_STATES)
        num_running = int(df_user.loc[is_running,'total_slots'].sum())
        num_qwaiting = int(df_user.loc[is_qwaiting,'total_slots'].sum())
        num_error = int(df_user.loc[is_error,'total_slots'].sum())
        print('# of running job tasks (estimated from squeue): {}'.format(num_running))
        print('# of queued job tasks (estimated from squeue): {}'.format(num_qwaiting))
        print('# of failed/cancelled job tasks (estimated from squeue): {}'.format(num_error))
        num_estimated_rows = int(df_user['task_count_estimated'].sum())
        if num_estimated_rows>0:
            txt = 'Note: {} row(s) had truncated/irregular SLURM array IDs; task counts are estimated.'
            print(txt.format(num_estimated_rows))
        print('')
        return
    is_running = df_user['state'].str.contains('r', regex=False)
    is_qwaiting = df_user['state'].str.contains('qw', regex=False)
    is_error = df_user['state'].str.contains('E', regex=False)
    num_running = df_user.loc[is_running,'total_slots'].sum()
    num_qwaiting = df_user.loc[is_qwaiting,'total_slots'].sum()
    num_error = df_user.loc[is_error,'total_slots'].sum()
    print('# of CPUs in use for running jobs: {}'.format(num_running))
    print('# of requested CPUs for queued jobs: {}'.format(num_qwaiting))
    print('# of CPUs for queued/running jobs in error: {}'.format(num_error))
    print('')

def get_scheduler_from_command(stat_command):
    try:
        command = shlex.split(stat_command)
    except ValueError:
        return None
    if len(command)==0:
        return None
    executable = os.path.basename(command[0])
    if executable=='qstat':
        return 'uge'
    if executable=='squeue':
        return 'slurm'
    return None

def _has_squeue_format_option(command):
    for token in command[1:]:
        if token in ['-o', '-O', '--format', '--Format']:
            return True
        if token.startswith('--format=') or token.startswith('--Format='):
            return True
        if token.startswith('-o') and token!='-o':
            return True
        if token.startswith('-O') and token!='-O':
            return True
    return False

def _has_squeue_noheader_option(command):
    for token in command[1:]:
        if token in ['-h', '--noheader']:
            return True
        if token.startswith('--noheader='):
            return True
    return False

def get_squeue_command_for_parsing(stat_command):
    try:
        command = shlex.split(stat_command)
    except ValueError:
        return stat_command
    if len(command)==0:
        return stat_command
    executable = os.path.basename(command[0])
    if executable!='squeue':
        return stat_command
    has_format_option = _has_squeue_format_option(command)
    has_noheader_option = _has_squeue_noheader_option(command)
    if not has_noheader_option:
        command.append('-h')
    if not has_format_option:
        command.extend(['-o', SLURM_SQUEUE_PARSE_FIELDS])
    return ' '.join([shlex.quote(item) for item in command])

def get_command_stdout_lines(command_str, example_file='', allow_failure=False, command_name='command', quiet_failure=False):
    if example_file != '':
        try:
            with open(example_file) as f:
                return f.readlines()
        except OSError as e:
            if not quiet_failure:
                print('Failed to read example file for {}: {}'.format(command_name, example_file))
                print(str(e))
            if allow_failure:
                return None
            exit(1)
    try:
        command = shlex.split(command_str)
    except ValueError as e:
        if not quiet_failure:
            print('Failed to parse {}: {}'.format(command_name, command_str))
            print(str(e))
        if allow_failure:
            return None
        exit(1)
    if len(command)==0:
        if not quiet_failure:
            print('Failed to run {}: command is empty'.format(command_name))
        if allow_failure:
            return None
        exit(1)
    try:
        command_out = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as e:
        if not quiet_failure:
            print('Failed to run {}: {}'.format(command_name, command_str))
            print(str(e))
        if allow_failure:
            return None
        exit(1)
    if command_out.returncode!=0:
        if not quiet_failure:
            print('Failed to run {}: {}'.format(command_name, command_str))
            command_stderr = command_out.stderr.decode('utf8')
            if command_stderr!='':
                print(command_stderr)
        if allow_failure:
            return None
        exit(1)
    command_stdout = command_out.stdout.decode('utf8')
    return command_stdout.split('\n')

def get_df(args):
    scheduler = get_scheduler_from_command(args.stat_command)
    if scheduler is None:
        print('Exiting. --stat_command does not support: {}'.format(args.stat_command))
        exit(1)
    if scheduler=='slurm':
        squeue_command = get_squeue_command_for_parsing(args.stat_command)
        lines = get_command_stdout_lines(command_str=squeue_command,
                                         example_file=args.example_file,
                                         allow_failure=False,
                                         command_name='--stat_command')
        df_user = get_squeue_user_df(lines)
        print_queued_job_summary(df_user, scheduler='slurm')
        partition_lines = get_command_stdout_lines(command_str=args.slurm_partition_command,
                                                   example_file=args.slurm_partition_example_file,
                                                   allow_failure=True,
                                                   command_name='--slurm_partition_command',
                                                   quiet_failure=True)
        partition_state_map = None
        if partition_lines is not None:
            df_partition = get_scontrol_partition_df(partition_lines)
            if df_partition.shape[0]>0:
                partition_state_map = df_partition.set_index('partition_name')['partition_state'].to_dict()
        node_lines = get_command_stdout_lines(command_str=args.slurm_node_command,
                                              example_file=args.slurm_node_example_file,
                                              allow_failure=True,
                                              command_name='--slurm_node_command')
        if node_lines is None:
            print('Skipping node resource summary because --slurm_node_command failed.')
            print('')
            return scheduler, None, df_user
        df_slurm_node = get_scontrol_node_df(node_lines, partition_state_map=partition_state_map)
        if df_slurm_node.shape[0]==0:
            print('Skipping node resource summary because SLURM node output could not be parsed.')
            print('Use --slurm_node_command "scontrol show node -o" or provide --slurm_node_example_file.')
            print('')
            return scheduler, None, df_user
        return scheduler, df_slurm_node, df_user
    if args.niter<1:
        print('Exiting. --niter must be >= 1 when using qstat mode.')
        exit(1)
    for i in range(args.niter):
        lines = get_command_stdout_lines(command_str=args.stat_command,
                                         example_file=args.example_file,
                                         allow_failure=False,
                                         command_name='--stat_command')
        df_i = get_qstat_df(lines)
        if i==0:
            df = df_i
            df_user = get_user_df(lines)
            print_queued_job_summary(df_user, scheduler='uge')
        else:
            df = _merge_qstat_iteration_min_availability(df, df_i)
    return scheduler, df, df_user

def adjust_ram_unit(df):
    for col in ['hc:mem_req','hl:mem_total']:
        raw = df[col].fillna('').astype(str).str.strip()
        units = raw.str.extract(r'([A-Za-z]+)$', expand=False).fillna('').str.upper()
        numeric_txt = raw.str.replace(r'[A-Za-z]+$', '', regex=True)
        df[col] = pandas.to_numeric(numeric_txt, errors='coerce').fillna(0.0).astype(float)
        df[col+'_unit'] = units
        is_t = (df[col+'_unit'].str.startswith('T')).fillna(False)
        if is_t.sum():
            df.loc[is_t,col] = df.loc[is_t,col] * 1000
            df.loc[is_t, col+'_unit'] = 'G'
        is_m = (df[col+'_unit'].str.startswith('M')).fillna(False)
        if is_m.sum():
            df.loc[is_m,col] = df.loc[is_m,col] * 0.001
            df.loc[is_m, col+'_unit'] = 'G'
        is_k = (df[col+'_unit'].str.startswith('K')).fillna(False)
        if is_k.sum():
            df.loc[is_k,col] = df.loc[is_k,col] * 0.000001
            df.loc[is_k, col+'_unit'] = 'G'
        is_g = (df[col+'_unit'].str.startswith('G')).fillna(False)
        if is_g.sum():
            df.loc[is_g, col+'_unit'] = 'G'
        is_unknown = (~is_t) & (~is_m) & (~is_k) & (~is_g)
        if is_unknown.sum():
            df.loc[is_unknown, col+'_unit'] = 'G'
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
    scheduler, df, df_user = get_df(args)
    if (scheduler=='slurm') and (df is None):
        print('Skipping cluster/node resource availability.')
        print('Reason: no parsed SLURM node data was available.')
        print('Provide --slurm_node_command or --slurm_node_example_file from "scontrol show node -o".')
        if args.out!='':
            df_user.to_csv(args.out, sep='\t', index=False)
        return
    df = adjust_ram_unit(df)
    print_cluster_summary(df)
    print_resource_availability(df, args)
    if args.out!='':
        df.to_csv(args.out, sep='\t', index=False)
