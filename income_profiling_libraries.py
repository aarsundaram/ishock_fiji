import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns_pal = sns.color_palette('Set1', n_colors=10, desat=.4)
greys_pal = sns.color_palette('Greys', n_colors=9)

def plot_income_profile_side_by_side(hh_df_in,split,xax='pcexp',district=None):

    # prep fig
    fig,axs = plt.subplots(2,2,sharex=True,gridspec_kw={'height_ratios': [1,3]})

    # set x limits    
    #ctrl = incprof_ctrl(district)
    #ctrl['bins'] = np.linspace((1E-3/12*hh_df_in['pcexp']).min(),ctrl['ul'],ctrl['nbins'])

    # loop over segments
    # dims = {'hoh_mainactivity_1':[(0,(hh_df_in['hoh_mainactivity']=='private_employee'),'private employees'),
    #                               (1,(hh_df_in['hoh_mainactivity']=='irregular/family'),'irregular & family workers')],
    #         'hoh_mainactivity_2':[(0,(hh_df_in['hoh_mainactivity']=='private_employee'),'private employees'),
    #                               (1,(hh_df_in['hoh_mainactivity']=='unemployed'),'unemployed')],
    #         'hoh_education':[(0,(hh_df_in['hoh_education']=='less_than_high_school'),'no high school diploma'),
    #                          (1,(hh_df_in['hoh_education']!='less_than_high_school'),'at least GCE/HS diploma')],
    #         'ethnicity':[(0,(hh_df_in['ethnicity']=='Sinhalese'),'Sinhalese'),
    #                      (1,(hh_df_in['ethnicity']!='Sinhalese'),'ethnic minorities')],
    #         'hhinc_mainsource_1':[(0,(hh_df_in['hhinc_mainsource']=='wages'),'wages'),
    #                             (1,(hh_df_in['hhinc_mainsource']=='other'),'other')],
    #         'hhinc_mainsource_2':[(0,(hh_df_in['hhinc_mainsource']=='wages'),'wages'),
    #                             (1,(hh_df_in['hhinc_mainsource']=='netirr'),'irregular earnings')],}

    dims = {'hoh_mainactivity_2' : [(0,(hh_df_in['hoh_mainactivity']=='Self employment/subsistence'),'self-employed'),
                                   (1,(hh_df_in['hoh_mainactivity']=='Not working'),'unemployed')],
            'hoh_education':[(0,(hh_df_in['hoh_education']=='Primary completion and below'),'no high school diploma'),
                           (1,(hh_df_in['hoh_education']!='At least some secondary completion'),'at least GCE/HS diploma')],
            'hhinc_mainsource_1':[(0,(hh_df_in['hhinc_mainsource_wage_other']=='inc_employ'),'wages'),
                                (1,(hh_df_in['hhinc_mainsource_wage_other']=='other'),'other')],
            'hhinc_mainsource_2':[(0,(hh_df_in['hhinc_mainsource_all']=='inc_employ'),'wages'),
                                (1,(hh_df_in['hhinc_mainsource_all']=='inc_gift'),'irregular earnings')]}


    for _sub,_slc,_lbl in dims[split]:
    # for _sub,_slc,_lbl in :

        # reload df
        hh_df = hh_df_in.copy()
        hh_df['pcexp'] *= 1E-3
        hh_df['pcwgt'] *= 1E-3
        # hh_df['hhwgt'] *= 1E-3

        # slice geographically
        if district is not None: 
            hh_df = hh_df.loc[hh_df['dist_name']==district,:].sort_values(xax,ascending=True)

        # get demographic slice
        hh_df = hh_df.loc[_slc,:].sort_values(xax,ascending=True)

        # cumulativepcwgt of the population
        hh_df['pop_cumfrac'] = hh_df['pcwgt'].cumsum()/hh_df['pcwgt'].sum()
        median = hh_df.loc[hh_df['pop_cumfrac']<=0.5,xax].max()

        # cumulativepcwgt of the poor
        hh_df['poor_cumfrac'] = hh_df.loc[hh_df['ispoor']==1,'pcwgt'].cumsum()/hh_df.loc[hh_df['ispoor']==1,'pcwgt'].sum()
        try: pov_rate = round(1E2*hh_df.loc[hh_df['ispoor']==1,'pcwgt'].sum()/hh_df['pcwgt'].sum(),1)
        except: pov_rate = '--'

        # poverty line
        pov_line = {}
        for _dec in [0.01*_ for _ in range(1,101)]:
            pov_line[_dec] = hh_df.loc[hh_df['poor_cumfrac']<=_dec,xax].max()

        # total income <-- use to structure plot
        tot_hgt, _bins = np.histogram(hh_df.eval(xax).clip(upper=1000),bins=20,weights=hh_df['pcwgt'])
        wid = (_bins[1]-_bins[0])

        # income_streams = ['hhinc_wages','hhinc_netirr',
        #                   'hhinc_cashaid',
        #                   #'hhinc_foodaid',
        #                   'hhinc_pension',
        #                   'hhinc_remits_int','hhinc_remits_dom',
        #                   'hhinc_capital','hhinc_windfall',
        #                   'hhinc_netag']

        income_streams = ['inc_employ', 'inc_commerce',
                        'inc_subsist', 'inc_remit', 
                        'inc_transfer', 'inc_gift', 
                        'inc_rent']

        is_hgt = {_i:[] for _i in income_streams}
        ratio = []
        for _i in income_streams:
            for n,b in enumerate(_bins): 
                try:
                    bin_slice = '({}>@b)&({}<=@_bins[@n+1])'.format(xax,xax)
                    is_hgt[_i].append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('1E2*pcwgt*{}'.format(_i)).sum()
                                            /hh_df.loc[hh_df.eval(bin_slice)].eval('pcwgt*inc_total').sum()))
                    if _i == income_streams[0]:
                        _num = 'pcinc'
                        ratio.append(float(hh_df.loc[hh_df.eval(bin_slice)].eval('pcwgt*{}'.format(_num)).sum()
                                           /hh_df.loc[hh_df.eval(bin_slice)].eval('pcwgt*{}'.format(xax)).sum()))
                except: pass

        _btm = [0 for _ in is_hgt['inc_employ']]
        is_format = {'inc_employ':('wages',sns_pal[0]),
                     'inc_commerce':('commerce',sns_pal[1]),
                     'inc_subsist':('subsistence',sns_pal[2]),
                     # 'hhinc_foodaid':('food aid',sns_pal[9]),
                     'inc_remit':('remittances',sns_pal[3]),
                     'inc_transfer':('transfer',sns_pal[4]),
                     'inc_gift':('gifts',sns_pal[5]),
                     'inc_rent':('rental income',sns_pal[6]),
                     #'hhinc_windfall':('windfalls',sns_pal[7]),
                     #'hhinc_netag':('agricultural',sns_pal[8])
                     }

        plt.sca(axs[0][_sub])
        plt.bar(_bins[:-1],tot_hgt,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=greys_pal[4],clip_on=False)

        if _sub == 0: plt.ylabel('Population [,000]',labelpad=10,fontsize=8)

        #plt.ylim(0,1000)

        #yticks = plt.gca().get_yticks() 
        #if _sub == 0: plt.yticks(yticks,[int(yt) if yt != 0 else '' for yt in yticks])
        #else: plt.yticks(yticks,['' for _ in yticks])

        for tick in axs[0][_sub].get_yticklabels():
            tick.set_verticalalignment('top')

        plt.gca().tick_params(axis='both', labelsize=8,length=0)
        plt.grid(True,axis='y',alpha=0.3)   

        plt.sca(axs[1][_sub])
        for _i in income_streams:
            plt.bar(_bins[:-1],is_hgt[_i],bottom=_btm,width=wid,align='edge',linewidth=0,alpha=0.6,facecolor=is_format[_i][1],label=is_format[_i][0])
            _btm = [i+j for i,j in zip(_btm,is_hgt[_i])]


        if xax == 'pcexp':
            plt.plot([pov_line[1.0],pov_line[1.0]],[0,135],lw=1.25,color=greys_pal[7],alpha=0.45,clip_on=False,ls=':')
            plt.annotate('{}%'.format(pov_rate),xy=(pov_line[1.0]-median*0.04,135),ha='right',va='top',annotation_clip=False,fontsize=8,weight='light',color=greys_pal[7],alpha=0.6)
            plt.annotate('poverty',xy=(pov_line[1.0]+median*0.04,135),ha='left',va='top',annotation_clip=False,fontsize=8,weight='light',color=greys_pal[7],alpha=0.6)
     
            plt.plot([median,median],[0,130],color=greys_pal[5],alpha=0.6,clip_on=False)
            plt.annotate('median',xy=(median*1.04,130),fontsize=8,color=greys_pal[7],ha='left',va='top',annotation_clip=False,weight='light',alpha=0.6)
        else:
            for _dec in [round(0.3*_,1) for _ in range(1,4)]:
                plt.plot([pov_line[_dec],pov_line[_dec]],[-1,135],lw=1.25,color=greys_pal[7],alpha=0.45,clip_on=False)
                _anno = '{}%{}'.format(int(1E2*_dec),' of poor' if _dec==0.3 else '')
                plt.annotate(_anno,xy=(pov_line[_dec],-2.5),ha='right',va='top',annotation_clip=False,fontsize=8,weight='light',rotation=30,color=greys_pal[7],alpha=0.6)

        xlabel_dict = {'pcinc':'income\n[$./cap/year]',
                       'pcexp':'consumption',
                       'pcinc_wages':'income from labor\n[$./cap/year]'}

        plt.xlabel(_lbl,labelpad=10,fontsize=8)

        #plt.xlim(0,20)
        #plt.xticks([5*_ for _ in range(1,int(20/5+1))])
        plt.ylim(0,100)

        plt.gca().tick_params(axis='both', labelsize=8,length=0)
        # plt.gca().tick_params(axis='y',va='top')

        plt.grid(True,axis='y',alpha=0.3)   

        if _sub == 0: 
            #plt.yticks([20*_ for _ in range(1,6)])
            for tick in axs[1][_sub].get_yticklabels():
                tick.set_verticalalignment('top')

            plt.ylabel('Contribution to total income [%]',labelpad=10,fontsize=8)
            plt.legend(loc='lower right',labelspacing=0.5,ncol=1,fontsize=6,borderpad=0.5,fancybox=True,frameon=True,framealpha=0.6)
        else:
            pass 
            #plt.yticks([20*_ for _ in range(1,6)],['' for _ in range(1,6)])

    sns.despine(left=True)

    fig.text(0.5,-0.0025,'Total consumption [,000 $/cap/year]',clip_on=False,va='top',ha='center',fontsize=8)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.04,wspace=0.05)
    #plt.plot([0,_ul],[0,0],color=greys_pal[4],lw=1)
    plt.savefig('figs/income_composition_{}_{}{}.pdf'.format(split,xax,'_'+district.lower().replace(' ','') if district is not None else ''),format='pdf',bbox_inches='tight')
    plt.savefig('figs/income_composition_{}_{}{}.png'.format(split,xax,'_'+district.lower().replace(' ','') if district is not None else ''),format='png',dpi=800,bbox_inches='tight')
    plt.close('all')
