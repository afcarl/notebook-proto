import re
import numpy as np
import pandas as pd
from operator import methodcaller
from collections import namedtuple
pd.set_option('display.multi_sparse', False)

class Exp(namedtuple('Exp', 'dataset model')):
    def col_name(self):
        return '%s' % (self.model)

class ResultsTable():
    def __init__(self, table):
        self.table = table

    @classmethod
    def from_table(cls, table, best='min'):
	def try_float(a):
	    try:
		return float(a)
	    except:
                if best == 'min':
                    return np.inf
                else:
                    return -np.inf
	    
	try_float_v = np.vectorize(try_float)
	res = try_float_v(table.as_matrix())

        if best == 'min':
            best_indices = np.argmin(res, axis=1)
        else:
            best_indices = np.argmax(res, axis=1)

        for i, best_index in enumerate(best_indices):
            res = table[table.columns[best_index]][table.index[i]]
            if '---' not in res:
                table[table.columns[best_index]][table.index[i]] = '<' + res + '>'

        return cls(table)
        

    @classmethod
    def from_data(cls, data, methods, exps, col_name, best='min', perc=True):
        if perc:
            make_res = np.vectorize(lambda mean, std: ('%.1f' % mean).lstrip('0') + '% +/- ' + ('%.1f' % std).lstrip('0') + '%')
            make_res_no_stdv = np.vectorize(lambda mean: ('%.1f' % mean).lstrip('0') + '%')
        else:
            make_res = np.vectorize(lambda mean, std: ('%.3f' % mean).lstrip('0') + ' +/- ' + ('%.3f' % std).lstrip('0'))
            make_res_no_stdv = np.vectorize(lambda mean: ('%.3f' % mean).lstrip('0'))

        # Get means and results
        if len(data.shape) == 3:
            means = data.mean(axis=2)
            stdvs = data.std(axis=2)
            if data.shape[2] == 1:
                res_table = make_res_no_stdv(means).tolist()
            else:
                res_table = make_res(means, stdvs).tolist()
        else:
            means = data
            stdvs = np.zeros(data.shape)
            res_table = make_res_no_stdv(means).tolist()
        
        # Determine which results are best
        for i, exp in enumerate(exps):
            exp_means = means[:, i].tolist()
            exp_stdvs = stdvs[:, i].tolist()
            
            if best == 'max':
                best_index = exp_means.index(max(exp_means))
                in_bounds = [j for j in range(0, len(exp_means))
                    if (exp_means[j] + exp_stdvs[j]) >= (exp_means[best_index] - exp_stdvs[best_index])]
            elif best == 'min':
                best_index = exp_means.index(min(exp_means))
                in_bounds = [j for j in range(0, len(exp_means))
                    if (exp_means[j] - exp_stdvs[j]) <= (exp_means[best_index] + exp_stdvs[best_index])]

            for j in in_bounds:
                res_table[j][i] = '<' + res_table[j][i] + '>'
                
        # Add dividers back in
        divider_indices = [i for i in range(len(methods)) if '---' in methods[i]]
        for divider_index in divider_indices:
            res_table.insert(divider_index, ['---'] * len(exps))
                
        table = pd.DataFrame(res_table, columns=map(methodcaller('col_name'), exps), index=methods)
        return cls(table)

    def to_latex(self, filename, replace_dict={}):  
        num_columns = len(self.table.columns)
        index_size = len(self.table.index.names)
        column_format = ' '.join(['c' for i in range(index_size)]) + ' ' + ' '.join(['c' for _ in range(num_columns)])
        with open(filename, 'w') as f:
            res = self.table.reset_index(level=range(len(self.table.index.names))).to_latex(column_format=column_format, index=False)
            res = re.sub('---.+\\\\', '\midrule', res)
            res = (res.replace('[', '\\specialcell{')
                .replace(']', '}')
                .replace(r'\$', '$')
                .replace(r'\_', '_')
                .replace(r'\{', '{')
                .replace(r'\}', '}')
                .replace('->', '\\textrightarrow')
                .replace('\\textbackslash', '\\')
                .replace('<', '{\\bf')
                .replace('>', '}')
                .replace('+/-', '$\\pm$')
            )
            for k, v in replace_dict.items():
                res = res.replace(k, v)
            f.write('\\resizebox{\\textwidth}{!}{%\n')
            f.write(res)
            f.write('}')
            f.write('\n')
