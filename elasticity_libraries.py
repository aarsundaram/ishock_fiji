import pandas as pd
import numpy as np 
from predictive_libraries import df_to_loglog_fit,array_to_linear_fit
np.random.seed()

from income_profiling_libraries import plot_income_profile_side_by_side
import matplotlib.pyplot as plt 
import seaborn as sns

sns_pal = sns.color_palette('Set1', n_colors=20, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

hhdf = pd.read_csv('./in/hhdf.csv')

incomes = ['inc_employ', 'inc_transfer', 'inc_commerce', 'inc_rent','inc_remit', 'inc_gift', 'inc_subsist']

incomes_dict = {'inc_employ':'wages',
    'inc_commerce':'selfemployed',
    'inc_subsist':'subsistence',
    'inc_remit':'remittances',
    'inc_transfer': 'transfer',
    'inc_gift':'windfall',
    'inc_rent': 'rental',
    'inc_total': 'total_income'
                }

#['At least some secondary completion','Primary completion and below','Post-secondary/tertiary and above']
educations_colors = {'Primary completion and below':sns_pal[0],
					 'At least some secondary completion':sns_pal[1],
					 'Post-secondary/tertiary and above':sns_pal[3]}
ethnicity_colors = {'Sinhalese':sns_pal[2],
					'Tamil':sns_pal[4],
					'other':sns_pal[5]}

#['Self employment/subsistence', 'Wage employment', 'Not working']
activity_colors = {'Wage employment':sns_pal[0],
				   'Self employment/subsistence':sns_pal[2],
				   #'employer':sns_pal[5],
				   #'irregular/family':sns_pal[6],
				   # 'family':sns_pal[7],
				   'Not working':sns_pal[8]}
#['inc_employ', 'inc_transfer', 'inc_commerce', 'inc_rent','inc_remit', 'inc_gift', 'inc_subsist']

msource_colors = {'inc_employ':sns_pal[0],
				  'inc_transfer':sns_pal[2],
				  'inc_commerce':sns_pal[5],
				  'inc_rent':sns_pal[6],
				  'inc_remit':sns_pal[4],
                  'inc_gift':sns_pal[7],
                  'inc_subsist': sns_pal[8]}

def plot_elasticity_regressions(nsims=10000,itoc='REL'):
	nsims_label = round(nsims/1e3,1)
	incomes = ['inc_employ', 'inc_transfer', 'inc_commerce', 'inc_rent','inc_remit', 'inc_gift', 'inc_subsist']

	income_shock = incomes

	####################################
	# load survey
	hh_df = pd.read_csv('./in/hhdf.csv')
	
	# hh_df = df[incomes].sum(level='hhid',axis=0)
	hh_df['popwgt'] = hh_df[['hhwgt','hhsize']].prod(axis=1)

	# hh_df[['hhsize','hhinc','pov_line','hhexp']] = df[['hhsize','hhinc','pov_line','hhexp']].mean(level='hhid')
	hh_df['hhinc_calc'] = hh_df[incomes].sum(axis=1)

	# group ethnicities
	#hh_df['ethnicity'] = hh_df['ethnicity'].replace({1:'Sinhalese',
													#  2:'Tamil',
													#  3:'Tamil',
													#  4:'other',
													#  5:'other',
													#  6:'other',
													#  9:'other'})

	#####################################
	# can't do this with hh with no income, or no consumption
	_ = hh_df['popwgt'].sum()
	hh_df = hh_df.loc[(hh_df['inc_total']>0)&(hh_df['total_cons']>0),:]
	_f = round(1E2*(_-hh_df['popwgt'].sum())/_,1)
	print('\nPAY ATTENTION HERE:\n-- dropping {}% of population b/c (inc<=0 | exp<=0)'.format(_f))

	if False:
		# hh_df['hhinc_wages'] = hh_df['hhinc_wages'].fillna(0)
		hh_df = hh_df.loc[hh_df['hhinc']>0,:]

		# print(round(1E2*hh_df.eval('(hhinc_cashaid/hhinc)*hhwgt').sum()/hh_df['hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']=='Sinhalese',:].eval('(hhinc_cashaid/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']=='Sinhalese','hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']!='Sinhalese',:].eval('(hhinc_cashaid/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']!='Sinhalese','hhwgt'].sum(),1))

		print(round(1E2*hh_df.loc[hh_df['ethnicity']=='Sinhalese',:].eval('(hhinc_remits_int/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']=='Sinhalese','hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']!='Sinhalese',:].eval('(hhinc_remits_int/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']!='Sinhalese','hhwgt'].sum(),1))

		print(round(1E2*hh_df.loc[hh_df['ethnicity']=='Sinhalese',:].eval('(hhinc_wages/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']=='Sinhalese','hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']!='Sinhalese',:].eval('(hhinc_wages/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']!='Sinhalese','hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']=='Sinhalese',:].eval('(hhinc_netirr/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']=='Sinhalese','hhwgt'].sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ethnicity']!='Sinhalese',:].eval('(hhinc_wages/hhinc)*hhwgt').sum()/hh_df.loc[hh_df['ethnicity']!='Sinhalese','hhwgt'].sum(),1))

		print(round(1E2*hh_df[['hhinc_pension','hhwgt']].prod(axis=1).sum()/hh_df[['hhinc','hhwgt']].prod(axis=1).sum(),1))
		print(round(1E2*hh_df.loc[hh_df['ispoor']==True,['hhinc_pension','hhwgt']].prod(axis=1).sum()/hh_df.loc[hh_df['ispoor']==True,['hhinc','hhwgt']].prod(axis=1).sum(),1))
		print(round(1E2*hh_df.loc[hh_df['hoh_education']=='less_than_high_school',['hhinc_pension','hhwgt']].prod(axis=1).sum()/hh_df.loc[hh_df['hoh_education']=='less_than_high_school',['hhinc','hhwgt']].prod(axis=1).sum(),1))
		print(round(1E2*hh_df.loc[hh_df['hoh_education']=='GCE/HS diploma',['hhinc_pension','hhwgt']].prod(axis=1).sum()/hh_df.loc[hh_df['hoh_education']=='GCE/HS diploma',['hhinc','hhwgt']].prod(axis=1).sum(),1))

	if False:

		incomes = ['hhinc_wages','hhinc_netag','hhinc_agsubsidy','hhinc_netirr',
			'hhinc_foodaid','hhinc_pension','hhinc_cashaid','hhinc_capital',
			'hhinc_remits_int','hhinc_remits_dom','hhinc_windfall']

		primfrac = ((hh_df[incomes].max(axis=1)/hh_df['hhinc'])*hh_df['popwgt']).sum()/hh_df['popwgt'].sum()
		print('primary source of income, fraction of total',primfrac)
		singlefrac = hh_df.loc[hh_df[incomes].max(axis=1)==hh_df['hhinc'],'popwgt'].sum()/hh_df['popwgt'].sum()
		print('fraction of hh with 1 source of income',singlefrac)

		hh_df = hh_df.loc[hh_df['ispoor']==True,:]
		primfrac = ((hh_df[incomes].max(axis=1)/hh_df['hhinc'])*hh_df['popwgt']).sum()/hh_df['popwgt'].sum()
		print('primary source of income, fraction of total among poor',primfrac)
		singlefrac = hh_df.loc[hh_df[incomes].max(axis=1)==hh_df['hhinc'],'popwgt'].sum()/hh_df['popwgt'].sum()
		print('fraction of poor hh with 1 source of income',singlefrac)
		assert(False)

		hh_df=hh_df.reset_index().set_index('hhinc_mainsource')
		print(hh_df['hhwgt'].sum(level='hhinc_mainsource')/hh_df['hhwgt'].sum())
		# print('% of poor households')
		# print(hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum(level='hhinc_mainsource')/hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum())
		print('\n% of households are poor')
		print(hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum(level='hhinc_mainsource')/hh_df['popwgt'].sum(level='hhinc_mainsource'))
		print('\npoor population')
		print((1E-3*hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum(level='hhinc_mainsource')).round(1))
		print((1E-3*hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum()).round(1))

		hh_df = hh_df.reset_index().set_index('hoh_mainactivity')		
		print('\npoor population')
		print((1E-3*hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum(level='hoh_mainactivity')).round(1))
		print((1E-3*hh_df.loc[hh_df['ispoor']==True,'popwgt'].sum()).round(1))
		assert(False)

	if False:
		for xax in ['pcexp','pcinc','pcinc_windfall','pcinc_capital','pcinc_netirr','pcinc_grossirr','pcinc_grossag','pcinc_wages','pcexp_food','pcexp_nonfood']:
			income_vs_consumption(hh_df.copy(),xax=xax,num='pcinc',dnm='pcexp')
		plot_income_profile_side_by_side(hh_df,split='ethnicity')
		plot_income_profile_side_by_side(hh_df,split='hoh_mainactivity_1')
		plot_income_profile_side_by_side(hh_df,split='hoh_mainactivity_2')
		plot_income_profile_side_by_side(hh_df,split='hoh_education')
		plot_income_profile_side_by_side(hh_df,split='hhinc_mainsource_1')
		plot_income_profile_side_by_side(hh_df,split='hhinc_mainsource_2')
		assert(False)


	# output df
	pr_cols = income_shock+[_+'_totalvalue' for _ in income_shock]+['pop_poverty','poverty_gap']
	poverty_record = pd.DataFrame(columns=pr_cols,index=range(nsims))

	# run simulations
	for nsim in range(nsims):
		if nsim%1E3==0: print('- {} of {}k nsims'.format(int(nsim/1E3),int(nsims/1E3)))

		for shock_chan in incomes: 
			hh_df[shock_chan+'_new'] = hh_df[shock_chan].copy() 

		for shock_chan in income_shock:
			
			# pull random value
			_rand = 1.+np.random.uniform(-0.1,0.1)
			poverty_record.loc[nsim,shock_chan] = _rand
			poverty_record.loc[nsim,shock_chan+'_totalvalue'] = _rand*(hh_df.eval('popwgt/hhsize')*hh_df[shock_chan]).sum()

			# adjust income
			hh_df[shock_chan+'_new'] = hh_df[shock_chan]*_rand


		# calculate new income
		hh_df['hhinc_new'] = hh_df[[_+'_new' for _ in incomes]].sum(axis=1)

		# calculate new consumption
		itoc_buffer = 0.9
		if itoc == 'REL': hh_df['pcexp_new'] = (hh_df['total_cons']/hh_df['hhsize'])*(hh_df['hhinc_new']/hh_df['hhinc_calc'])
		elif itoc == 'REL-': hh_df['pcexp_new'] = (hh_df['total_cons']/hh_df['hhsize'])*itoc_buffer*(hh_df['hhinc_new']/hh_df['hhinc_calc'])
		elif itoc == 'ABS': hh_df['pcexp_new'] = (hh_df['total_cons']+(1/0.9)*(hh_df['hhinc_new']-hh_df['hhinc_calc']))/hh_df['hhsize']
		elif itoc == 'ABS-': hh_df['pcexp_new'] = (hh_df['total_cons']+(itoc_buffer/0.9)*(hh_df['hhinc_new']-hh_df['hhinc_calc']))/hh_df['hhsize']
		else: assert(False)

		# check impoverishment & record results
		is_poor = hh_df['pcexp_new'] < hh_df['pov_line']
		poverty_record.loc[nsim,'pop_poverty'] = hh_df.loc[is_poor,'popwgt'].sum()
		poverty_record.loc[nsim,'poverty_gap'] = (hh_df.loc[is_poor,:].eval('-1*popwgt*(pcexp_new-pov_line)/pov_line').sum()/hh_df.loc[is_poor,'popwgt'].sum())

		# by education
		for ed in hh_df['hoh_education'].unique():
			is_poor_by_ed = (hh_df['pcexp_new']<hh_df['pov_line'])&(hh_df['hoh_education']==ed)
			poverty_record.loc[nsim,'pop_poverty_{}'.format(ed)] = hh_df.loc[is_poor_by_ed,'popwgt'].sum()
			poverty_record.loc[nsim,'poverty_gap_{}'.format(ed)] = hh_df.loc[is_poor_by_ed,:].eval('-1*popwgt*(pcexp_new-pov_line)/pov_line').sum()/hh_df.loc[is_poor_by_ed,'popwgt'].sum()

		# by ethnicity
		# for eth in hh_df['ethnicity'].unique():
		# 	is_poor_by_eth = (hh_df['pcexp_new']<hh_df['pov_line'])&(hh_df['ethnicity']==eth)
		# 	poverty_record.loc[nsim,'pop_poverty_{}'.format(eth)] = hh_df.loc[is_poor_by_eth,'popwgt'].sum()
		# 	poverty_record.loc[nsim,'poverty_gap_{}'.format(eth)] = hh_df.loc[is_poor_by_eth,:].eval('-1*popwgt*(pcexp_new-pov_line)/pov_line').sum()/hh_df.loc[is_poor_by_eth,'popwgt'].sum()	

		# by activity
		for job in hh_df['hoh_mainactivity'].unique():
			is_poor_by_job = (hh_df['pcexp_new']<hh_df['pov_line'])&(hh_df['hoh_mainactivity']==job)
			poverty_record.loc[nsim,'pop_poverty_{}'.format(job)] = hh_df.loc[is_poor_by_job,'popwgt'].sum()	
			poverty_record.loc[nsim,'poverty_gap_{}'.format(job)] = hh_df.loc[is_poor_by_job,:].eval('-1*popwgt*(pcexp_new-pov_line)/pov_line').sum()/hh_df.loc[is_poor_by_job,'popwgt'].sum()	

		# by main income source
		for ms in hh_df['hhinc_mainsource_all'].unique():
			is_poor_by_ms = (hh_df['pcexp_new']<hh_df['pov_line'])&(hh_df['hhinc_mainsource_all']==ms)
			poverty_record.loc[nsim,'pop_poverty_{}'.format(ms)] = hh_df.loc[is_poor_by_ms,'popwgt'].sum()	
			poverty_record.loc[nsim,'poverty_gap_{}'.format(ms)] = hh_df.loc[is_poor_by_ms,:].eval('-1*popwgt*(pcexp_new-pov_line)/pov_line').sum()/hh_df.loc[is_poor_by_ms,'popwgt'].sum()	

	# run regressions
	elasticities = pd.DataFrame(columns=['elasticity'],index=income_shock)
	elasticities.index.name='ichan'
	for shock_chan in income_shock:

		# general pop, log-log: % response to 1% income shock
		_,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'pop_poverty')
		elasticities.loc[shock_chan,'elasticity'] = abs(coeff)
		# general pop, level-log: unit response to % shock
		_,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan+'_totalvalue','pop_poverty',ylog=False)
		elasticities.loc[shock_chan,'head_elasticity'] = abs(1E-3*1E-2*coeff)# to thousands

		# poverty gap elasticity: % 
		_,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap')
		elasticities.loc[shock_chan,'elasticity_povertygap'] = abs(coeff)
		# poverty level-log elasticity
		_,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap',ylog=False)
		elasticities.loc[shock_chan,'pp_elasticity_povertygap'] = abs(coeff) # coeff = 1/100 units of y, * 100 = %


		####################################		
		# by HOH education
		for ed in hh_df['hoh_education'].unique():

			####################################
			# poverty by education, log-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'pop_poverty_{}'.format(ed))
			except: coeff = 0
			elasticities.loc[shock_chan,'elast_{}'.format(ed)] = abs(coeff)
			# poverty by education, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan+'_totalvalue','pop_poverty_{}'.format(ed),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'head_elast_{}'.format(ed)] = abs(1E-3*1E-2*coeff)

			####################################
			# poverty gap by education, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap_{}'.format(ed),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'pp_elast_povertygap_{}'.format(ed)] = abs(coeff)

		####################################
		# by ethnicity
		# for eth in hh_df['ethnicity'].unique():

		# 	####################################
		# 	# by ethnicity, log-log
		# 	try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'pop_poverty_{}'.format(eth))
		# 	except: coeff = 0
		# 	elasticities.loc[shock_chan,'elast_{}'.format(eth)] = abs(coeff)
		# 	# by ethnicity, level-LOG
		# 	try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan+'_totalvalue','pop_poverty_{}'.format(eth),ylog=False)
		# 	except: coeff = 0
		# 	elasticities.loc[shock_chan,'head_elast_{}'.format(eth)] = abs(1E-3*1E-2*coeff)

		# 	####################################
		# 	# poverty gap by ethnicity, level-log
		# 	try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap_{}'.format(eth),ylog=False)
		# 	except: coeff = 0
		# 	elasticities.loc[shock_chan,'pp_elast_povertygap_{}'.format(eth)] = abs(coeff)


		####################################
		# by HOH activity
		for job in hh_df['hoh_mainactivity'].unique():

			####################################
			# poverty by activity, log-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'pop_poverty_{}'.format(job))
			except: coeff = 0
			elasticities.loc[shock_chan,'elast_{}'.format(job)] = abs(coeff)
			# poverty by activity, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan+'_totalvalue','pop_poverty_{}'.format(job),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'head_elast_{}'.format(job)] = abs(1E-3*1E-2*coeff)

			####################################
			# poverty gap by ethnicity, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap_{}'.format(job),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'pp_elast_povertygap_{}'.format(job)] = abs(coeff)


		####################################
		# by main source of income
		for ms in hh_df['hhinc_mainsource_all'].unique():

			####################################
			# poverty by activity, log-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'pop_poverty_{}'.format(ms))
			except: coeff = 0
			elasticities.loc[shock_chan,'elast_{}'.format(ms)] = abs(coeff)
			# poverty by activity, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan+'_totalvalue','pop_poverty_{}'.format(ms),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'head_elast_{}'.format(ms)] = abs(1E-3*1E-2*coeff)

			####################################
			# poverty gap by ethnicity, level-log
			try: _,coeff,_ = df_to_loglog_fit(poverty_record,shock_chan,'poverty_gap_{}'.format(ms),ylog=False)
			except: coeff = 0
			elasticities.loc[shock_chan,'pp_elast_povertygap_{}'.format(ms)] = abs(coeff)

	######################################
	# PLOT 1a: loglog/relative elasticity of poverty, by education
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('elasticity',ascending=True).reset_index()

		# plot output
		plt.scatter(elasticities['elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for ed in hh_df['hoh_education'].unique():
			if 'elast_{}'.format(ed) == 'elast_nan': continue 
			plt.scatter(elasticities['elast_{}'.format(ed)],elasticities.index,clip_on=False,zorder=98,label=ed.replace('_',' '),s=20,alpha=0.7,color=educations_colors[ed])

		# connect vertically
		for ed in hh_df['hoh_education'].unique():
			if 'elast_{}'.format(ed) == 'elast_nan': continue
			plt.plot(elasticities['elast_{}'.format(ed)],elasticities.index,color=educations_colors[ed],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('educational attainment',prop={'size':8})
		plt.xlabel('Elasticity of poverty incidence to income shocks\n[% response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlog-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_incidence_by_education.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_incidence_by_education.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')

	######################################
	# PLOT 1b: loglog/relative elasticity of poverty, by ethnicity
	# sort output
	# if True:
	# 	plt.figure(figsize=(8,4))
	# 	elasticities = elasticities.sort_values('elasticity',ascending=True).reset_index(drop=True)

	# 	# plot output
	# 	plt.scatter(elasticities['elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
	# 	for eth in hh_df['ethnicity'].unique():
	# 		if 'elast_{}'.format(eth) == 'elast_nan': continue 
	# 		plt.scatter(elasticities['elast_{}'.format(eth)],elasticities.index,clip_on=False,zorder=98,label=eth,s=20,alpha=0.7,color=ethnicity_colors[eth])

	# 	# connect vertically
	# 	for eth in hh_df['ethnicity'].unique():
	# 		plt.plot(elasticities['elast_{}'.format(eth)],elasticities.index,color=ethnicity_colors[eth],lw=0.5,zorder=90,label=None,alpha=0.3)

	# 	# format plot
	# 	plt.xlim(0)
	# 	plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
	# 	plt.gca().tick_params(axis='both', labelsize=8,length=0)
	# 	leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
	# 	leg.set_title('ethnicity',prop={'size':8})
	# 	plt.xlabel('Elasticity of poverty incidence to income shocks\n[% response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
	# 	plt.annotate('Source: Sri Lanka HIES (2016)\nlog-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
	# 	plt.grid(axis='x',alpha=0.2)

	# 	sns.despine()
	# 	plt.savefig('figs/elasticities_incidence_by_ethnicity.pdf',format='pdf',bbox_inches='tight')
	# 	plt.savefig('figs/elasticities_incidence_by_ethnicity.png',format='png',dpi=800,bbox_inches='tight')
	# 	plt.close('all')   

	######################################
	# PLOT 1c: loglog/relative elasticity of poverty, by job status
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('elasticity',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for job in hh_df['hoh_mainactivity'].unique():
			if 'elast_{}'.format(job) == 'elast_nan': continue 
			plt.scatter(elasticities['elast_{}'.format(job)],elasticities.index,clip_on=False,zorder=98,label=job.replace('_',' '),s=20,alpha=0.7,color=activity_colors[job])

		# connect vertically
		for job in hh_df['hoh_mainactivity'].unique():
			plt.plot(elasticities['elast_{}'.format(job)],elasticities.index,color=activity_colors[job],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('employment status',prop={'size':8})
		plt.xlabel('Elasticity of poverty incidence to income shocks\n[% response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlog-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_incidence_by_activity.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_incidence_by_activity.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   

	######################################
	# PLOT 1d: loglog/relative elasticity of poverty, by main source of income
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('elasticity',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for ms in hh_df['hhinc_mainsource_all'].unique():
			if 'elast_{}'.format(ms) == 'elast_nan': continue 
			plt.scatter(elasticities['elast_{}'.format(ms)],elasticities.index,clip_on=False,zorder=98,label=incomes_dict[ms],s=20,alpha=0.7,color=msource_colors[ms])

		# connect vertically
		for ms in hh_df['hhinc_mainsource_all'].unique():
			plt.plot(elasticities['elast_{}'.format(ms)],elasticities.index,color=msource_colors[ms],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('main income source',prop={'size':8})
		plt.xlabel('Elasticity of poverty incidence to income shocks\n[% response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlog-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_incidence_by_mainincome.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_incidence_by_mainincome.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   


	######################################
	# PLOT 2a: levellog/absolute elasticity of poverty headcount, by education
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('head_elasticity',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['head_elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for ed in hh_df['hoh_education'].unique():
			if 'head_elast_{}'.format(ed) == 'head_elast_nan': continue 
			plt.scatter(elasticities['head_elast_{}'.format(ed)],elasticities.index,clip_on=False,zorder=98,label=ed.replace('_',' '),s=20,alpha=0.7,color=educations_colors[ed])

		# connect vertically
		for ed in hh_df['hoh_education'].unique():
			plt.plot(elasticities['head_elast_{}'.format(ed)],elasticities.index,color=educations_colors[ed],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('educational attainment',prop={'size':8})
		plt.xlabel('Elasticity of poverty headcount to income shocks\n[,000 individuals impoverished by 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_headcount_by_education.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_headcount_by_education.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')  

	######################################
	# PLOT 2b: levellog/absolute elasticity of poverty headcount, by ethnicity
	# sort output
	# if True:
	# 	plt.figure(figsize=(8,4))
	# 	elasticities = elasticities.sort_values('head_elasticity',ascending=True).reset_index(drop=True)

	# 	# plot output
	# 	plt.scatter(elasticities['head_elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
	# 	for eth in hh_df['ethnicity'].unique():
	# 		if 'head_elast_{}'.format(eth) == 'head_elast_nan': continue 
	# 		plt.scatter(elasticities['head_elast_{}'.format(eth)],elasticities.index,clip_on=False,zorder=98,label=eth,s=20,alpha=0.7,color=ethnicity_colors[eth])

	# 	# connect vertically
	# 	for eth in hh_df['ethnicity'].unique():
	# 		plt.plot(elasticities['head_elast_{}'.format(eth)],elasticities.index,color=ethnicity_colors[eth],lw=0.5,zorder=90,label=None,alpha=0.3)

	# 	# format plot
	# 	plt.xlim(0)
	# 	plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
	# 	plt.gca().tick_params(axis='both', labelsize=8,length=0)
	# 	leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
	# 	leg.set_title('ethnicity',prop={'size':8})
	# 	plt.xlabel('Elasticity of poverty headcount to income shocks\n[,000 individuals impoverished by 1% shock]',fontsize=8,linespacing=2,labelpad=10)
	# 	plt.annotate('Source: Sri Lanka HIES (2016)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
	# 	plt.grid(axis='x',alpha=0.2)

	# 	sns.despine()
	# 	plt.savefig('figs/elasticities_headcount_by_ethnicity.pdf',format='pdf',bbox_inches='tight')
	# 	plt.savefig('figs/elasticities_headcount_by_ethnicity.png',format='png',dpi=800,bbox_inches='tight')
	# 	plt.close('all')   

	######################################
	# PLOT 2c: levellog/absolute elasticity of poverty headcount, by job status
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('head_elasticity',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['head_elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for job in hh_df['hoh_mainactivity'].unique():
			if 'head_elast_{}'.format(job) == 'head_elast_nan': continue 
			plt.scatter(elasticities['head_elast_{}'.format(job)],elasticities.index,clip_on=False,zorder=98,label=job.replace('_',' '),s=20,alpha=0.7,color=activity_colors[job])

		# connect vertically
		for job in hh_df['hoh_mainactivity'].unique():
			plt.plot(elasticities['head_elast_{}'.format(job)],elasticities.index,color=activity_colors[job],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('employment status',prop={'size':8})
		plt.xlabel('Elasticity of poverty headcount to income shocks\n[,000 individuals impoverished by 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_headcount_by_activity.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_headcount_by_activity.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   

	######################################
	# PLOT 2d: levellog/absolute elasticity of poverty headcount, by main source of income
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('head_elasticity',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['head_elasticity'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for ms in hh_df['hhinc_mainsource_all'].unique():
			if 'head_elast_{}'.format(ms) == 'head_elast_nan': continue 
			plt.scatter(elasticities['head_elast_{}'.format(ms)],elasticities.index,clip_on=False,zorder=98,label=incomes_dict[ms],s=20,alpha=0.7,color=msource_colors[ms])

		# connect vertically
		for ms in hh_df['hhinc_mainsource_all'].unique():
			plt.plot(elasticities['head_elast_{}'.format(ms)],elasticities.index,color=msource_colors[ms],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('main income source',prop={'size':8})
		plt.xlabel('Elasticity of poverty headcount to income shocks\n[,000 individuals impoverished by 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_headcount_by_mainincome.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_headcount_by_mainincome.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   

	######################################
	# PLOT 3a: levellog/relative elasticity of poverty gap, by education
	# sort output
	# if True:
	# 	plt.figure(figsize=(8,4))
	# 	elasticities = elasticities.sort_values('pp_elasticity_povertygap',ascending=True).reset_index(drop=True)

	# 	# plot output
	# 	plt.scatter(elasticities['pp_elasticity_povertygap'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
	# 	for ed in hh_df['hoh_education'].unique():
	# 		if 'elast_{}'.format(ed) == 'elast_nan': continue 
	# 		plt.scatter(elasticities['pp_elast_povertygap_{}'.format(ed)],elasticities.index,clip_on=False,zorder=98,label=ed.replace('_',' '),s=20,alpha=0.7,color=educations_colors[ed])

	# 	# connect vertically
	# 	for ed in hh_df['hoh_education'].unique():
	# 		plt.plot(elasticities['pp_elast_povertygap_{}'.format(ed)],elasticities.index,color=educations_colors[ed],lw=0.5,zorder=90,label=None,alpha=0.3)

	# 	# format plot
	# 	plt.xlim(0)
	# 	plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
	# 	plt.gca().tick_params(axis='both', labelsize=8,length=0)
	# 	leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
	# 	leg.set_title('educational attainment',prop={'size':8})
	# 	plt.xlabel('Elasticity of poverty gap to income shocks\n[pp response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
	# 	plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
	# 	plt.grid(axis='x',alpha=0.2)

	# 	sns.despine()
	# 	plt.savefig('figs/elasticities_povgap_by_education.pdf',format='pdf',bbox_inches='tight')
	# 	plt.savefig('figs/elasticities_povgap_by_education.png',format='png',dpi=800,bbox_inches='tight')
	# 	plt.close('all')  

	######################################
	# PLOT 3b: levellog/relative elasticity of poverty gap, by ethnicity
	# sort output
	# if True:
	# 	plt.figure(figsize=(8,4))
	# 	elasticities = elasticities.sort_values('pp_elasticity_povertygap',ascending=True).reset_index(drop=True)

	# 	# plot output
	# 	plt.scatter(elasticities['pp_elasticity_povertygap'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
	# 	for eth in hh_df['ethnicity'].unique():
	# 		if 'elast_{}'.format(eth) == 'elast_nan': continue 
	# 		plt.scatter(elasticities['pp_elast_povertygap_{}'.format(eth)],elasticities.index,clip_on=False,zorder=98,label=eth,s=20,alpha=0.7,color=ethnicity_colors[eth])

	# 	# connect vertically
	# 	for eth in hh_df['ethnicity'].unique():
	# 		plt.plot(elasticities['pp_elast_povertygap_{}'.format(eth)],elasticities.index,color=ethnicity_colors[eth],lw=0.5,zorder=90,label=None,alpha=0.3)

	# 	# format plot
	# 	plt.xlim(0)
	# 	plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
	# 	plt.gca().tick_params(axis='both', labelsize=8,length=0)
	# 	leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
	# 	leg.set_title('ethnicity',prop={'size':8})
	# 	plt.xlabel('Elasticity of poverty gap to income shocks\n[pp response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
	# 	plt.annotate('Source: Sri Lanka HIES (2016)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
	# 	plt.grid(axis='x',alpha=0.2)

	# 	sns.despine()
	# 	plt.savefig('figs/elasticities_povgap_by_ethnicity.pdf',format='pdf',bbox_inches='tight')
	# 	plt.savefig('figs/elasticities_povgap_by_ethnicity.png',format='png',dpi=800,bbox_inches='tight')
	# 	plt.close('all')   

	######################################
	# PLOT 3c: levellog/relative elasticity of poverty gap, by job status
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('pp_elasticity_povertygap',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['pp_elasticity_povertygap'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for job in hh_df['hoh_mainactivity'].unique():
			if 'elast_{}'.format(job) == 'elast_nan': continue 
			plt.scatter(elasticities['pp_elast_povertygap_{}'.format(job)],elasticities.index,clip_on=False,zorder=98,label=job.replace('_',' '),s=20,alpha=0.7,color=activity_colors[job])

		# connect vertically
		for job in hh_df['hoh_mainactivity'].unique():
			plt.plot(elasticities['pp_elast_povertygap_{}'.format(job)],elasticities.index,color=activity_colors[job],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('employment status',prop={'size':8})
		plt.xlabel('Elasticity of poverty gap to income shocks\n[pp response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_povgap_by_activity.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_povgap_by_activity.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   

	######################################
	# PLOT 3d: levellog/absolute elasticity of poverty headcount, by main source of income
	# sort output
	if True:
		plt.figure(figsize=(8,4))
		elasticities = elasticities.sort_values('pp_elasticity_povertygap',ascending=True).reset_index(drop=True)

		# plot output
		plt.scatter(elasticities['pp_elasticity_povertygap'],elasticities.index,clip_on=False,zorder=99,label='all groups',marker='x',s=35,color=greys_pal[7],alpha=0.7)
		for ms in hh_df['hhinc_mainsource_all'].unique():
			if 'head_elast_{}'.format(ms) == 'head_elast_nan': continue 
			plt.scatter(elasticities['pp_elast_povertygap_{}'.format(ms)],elasticities.index,clip_on=False,zorder=98,label=incomes_dict[ms],s=20,alpha=0.7,color=msource_colors[ms])

		# connect vertically
		for ms in hh_df['hhinc_mainsource_all'].unique():
			plt.plot(elasticities['pp_elast_povertygap_{}'.format(ms)],elasticities.index,color=msource_colors[ms],lw=0.5,zorder=90,label=None,alpha=0.3)

		# format plot
		plt.xlim(0)
		plt.yticks(elasticities.index,[incomes_dict[_] for _ in elasticities['ichan'].values])
		plt.gca().tick_params(axis='both', labelsize=8,length=0)
		leg = plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=8,borderpad=0.75,fancybox=True,frameon=True,framealpha=0.9)
		leg.set_title('main income source',prop={'size':8})
		plt.xlabel('Elasticity of poverty gap to income shocks\n[pp response to 1% shock]',fontsize=8,linespacing=2,labelpad=10)
		plt.annotate('Source: Fiji HIES (2019-20)\nlevel-log regression ({}k sims)'.format(nsims_label),xy=(0.99,1.01),xycoords='axes fraction',fontsize=8,color=greys_pal[5],annotation_clip=False,ha='right',va='bottom')
		plt.grid(axis='x',alpha=0.2)

		sns.despine()
		plt.savefig('figs/elasticities_povgap_by_mainincome.pdf',format='pdf',bbox_inches='tight')
		plt.savefig('figs/elasticities_povgap_by_mainincome.png',format='png',dpi=800,bbox_inches='tight')
		plt.close('all')   

