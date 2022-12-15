#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MultiplexMS Gooey-based graphical user interface
"""

from gooey import Gooey, GooeyParser, local_resource_path
import multiplex_ms_utils as mms


@Gooey(optional_cols=2,
       program_name="MultiplexMS - Companion tool",
       program_description='Create multiplexed grids and '
                           'deconvolute HR-MS data from pooled samples',
       show_failure_modal=False,
       image_dir=local_resource_path('gui_images'),
       default_size=(800, 630),
       menu=[{
           'name': 'Help',
           'items': [
               {
                'type': 'Link',
                'menuTitle': 'Documentation',
                'url': 'https://liningtonlab.github.io/MultiplexMS_documentation/'
               },
               {'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': 'MultiplexMS - Companion tool',
                'description': 'If you use the workflow, please cite MultiplexMS: \n'
                               'Paper title, Journal, year, doi',
                'version': '1.0 - presubmission',
                'developer': 'Michael J.J. Recchia\n'
                             'Tim U.H. Baumeister\n'
                             'Roger G. Linington',
                'license': 'MIT'
                }
           ]
       }]
       )
def main():

    # # # Initial Gooey setup and settings
    parser = GooeyParser()
    subs = parser.add_subparsers(help='functions', dest='function')

    # # # User table creation: Preparation tables and initial and rearranged grids
    preparation_parser = subs.add_parser('Preparation',
                                         help='Create MultiplexMS grids and preparation tables necessary for the'
                                              'MultiplexMS workflow')
    preparation_parser.add_argument('file_path',
                                    metavar='Path',
                                    help='Specify path to the .csv or .txt file which contains the sample names',
                                    type=str, widget='FileChooser',
                                    gooey_options={
                                        'validator': {
                                            'test': 'any([user_input.split(".")[-1] == ext for ext in ["csv", "txt"]])',
                                            'message': 'File needs to be a .csv or .txt file'
                                        }
                                    }
                                    )
    preparation_parser.add_argument('out_path',
                                    metavar='File destination',
                                    help='Specify folder where .csv tables shall be saved',
                                    type=str, widget='DirChooser')
    preparation_parser.add_argument('grid_side_length_r',
                                    metavar='Grid side length r', default=10, type=int,
                                    help='Side length r of the desired square grid '
                                         '(default is 10 for a grid-size of a 100)',
                                    gooey_options={
                                        'validator': {
                                            'test': 'all([int(user_input) > 0, '
                                                    'float(user_input) % int(user_input) == 0])',
                                            'message': 'The grid side length must be a positive integer value'
                                        }
                                    }
                                    )
    preparation_parser.add_argument('--randomize',
                                    metavar='Randomize',
                                    help='Randomize sample list (specify seed for reproducibility)',
                                    widget='BlockCheckbox', action='store_true')
    preparation_parser.add_argument('--seed',
                                    metavar='Seed',
                                    help='Seed (integer value) that determines the outcome of the randomization of '
                                         'sample names',
                                    type=int,
                                    gooey_options={
                                        'validator': {
                                            'test': 'all([int(user_input) > 0, '
                                                    'float(user_input) % int(user_input) == 0])',
                                            'message': 'The seed must be a positive integer value'
                                        }
                                    }
                                    )
    preparation_parser.add_argument('--experiment-tag',
                                    metavar='Experiment tag',
                                    help='Optional tag that will be added to the exported .csv files',
                                    gooey_options={
                                        'validator': {
                                            'test': 'all([i not in """"\'!@#$%^&*().+?:;=,<>""" for i in user_input])',
                                            'message': 'Must not contain special characters - !@#$%^&*().+?:;=,<>/'
                                        }
                                    }
                                    )

    #########################################################

    deconvolution_parser = subs.add_parser(
        'Deconvolution', help='Deconvolute pooled samples feature table to obtain feature table that contains'
                              ' each sample')
    deconvolution_parser.add_argument('grids',
                                      metavar='Folder with initial and rearranged grid(s)',
                                      help='Select folder with grid files',
                                      type=str, widget='DirChooser'
                                      )
    deconvolution_parser.add_argument('feature_presence_table',
                                      metavar='Pooled samples feature table',
                                      help='Select feature table (.csv) with pooled samples',
                                      type=str, widget='FileChooser',
                                      gooey_options={
                                          'validator': {
                                              'test': 'user_input.split(".")[-1] == "csv"',
                                              'message': 'File needs to be a .csv file'
                                          }
                                      }
                                      )
    deconvolution_parser.add_argument('feature_table_format',
                                      metavar='Choose format of feature table',
                                      help='Are samples oriented in columns or rows?',
                                      choices=['Samples in columns', 'Samples in rows'],
                                      type=str, widget='Dropdown',
                                      default='Samples in columns'
                                      )
    deconvolution_parser.add_argument('out_path',
                                      metavar='File destination',
                                      help='Select folder where the deconvoluted feature table shall be saved',
                                      type=str, widget='DirChooser')

    #########################################################

    cleaning_parser = subs.add_parser(
        'Cleaning', help='Remove absent features from feature table after the deconvolution step.'
                         'In addition, remove features that are present in more than x% of the samples.')
    cleaning_parser.add_argument('feature_presence_table',
                                 metavar='Deconvoluted feature table',
                                 help='Select deconvoluted feature table (.csv)',
                                 type=str, widget='FileChooser',
                                 gooey_options={
                                     'validator': {
                                         'test': 'user_input.split(".")[-1] == "csv"',
                                         'message': 'File needs to be a .csv file'
                                     }
                                 }
                                 )
    cleaning_parser.add_argument('out_path',
                                 metavar='File destination',
                                 help='Select folder where the cleaned deconvoluted feature table shall be saved',
                                 type=str, widget='DirChooser'
                                 )
    cleaning_parser.add_argument('feature_table_format',
                                 metavar='Choose format of feature table',
                                 help='Are samples oriented in columns or rows?',
                                 choices=['Samples in columns', 'Samples in rows'],
                                 type=str, widget='Dropdown',
                                 default='Samples in columns'
                                 )
    cleaning_parser.add_argument('--critical_threshold',
                                 metavar='Critical threshold',
                                 help='Features that are present in more than (critical threshold) % '
                                      'of the samples are removed from the feature table',
                                 type=int,
                                 gooey_options={
                                     'validator': {
                                         'test': 'all([float(user_input) <= 100, float(user_input) > 0])',
                                         'message': 'Critical threshold needs to be a number between 0 and 100'
                                     }
                                 }
                                 )

    #########################################################

    args = parser.parse_args()

    # # Generation of grids from sample list and preparation table
    if args.function == 'Preparation':

        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
        sample_names = mms.read_sample_list(args.file_path)

        if len(sample_names) == 0:
            raise ValueError('The sample list .csv is empty')

        mms.create_user_tables_output(sample_names=sample_names,
                                      rel_path=args.out_path,
                                      r=args.grid_side_length_r,
                                      seed=args.seed,
                                      randomize_grids=args.randomize,
                                      experiment_tag=args.experiment_tag)
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

    # # Deconvolution of feature table containing pooled samples to feature table with all samples
    if args.function == 'Deconvolution':

        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
        print('Reading initial and rearranged grids')
        combined_grids = mms.combine_grids(args.grids)

        print(f'Reading the feature ({args.feature_table_format}) table and binarize it')
        samples_in_columns = args.feature_table_format == 'Samples in columns'
        fpt = mms.prepare_user_feature_presence_table(args.feature_presence_table,
                                                      samples_in_columns)

        print('Verifying that pooled sample names match between grids and the feature table')
        mms.verify_pooled_sample_names(feature_presence_table=fpt,
                                       path_to_grids=args.grids)

        print('Deconvoluting the pooled samples feature table')
        mms.batch_deconvolute(path_to_grids=args.grids,
                              combined_grids=combined_grids,
                              feature_presence_table=fpt,
                              file_destination=args.out_path)
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')

    if args.function == 'Cleaning':

        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
        print(f'Reading deconvoluted feature table ({args.feature_table_format}) table')
        samples_in_columns = args.feature_table_format == 'Samples in columns'
        fpt = mms.prepare_user_feature_presence_table(args.feature_presence_table,
                                                      samples_in_columns)

        print('Checking deconvoluted feature table')
        mms.remove_features_from_table(feature_presence_table=fpt,
                                       critical_percentage=args.critical_threshold,
                                       file_path=args.feature_presence_table,
                                       out_path=args.out_path)
        print('-+-+-+-+-+-+-+-+-+-+-+-+-+-+-')


if __name__ == '__main__':
    main()
