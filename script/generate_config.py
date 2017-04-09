#!/usr/bin/env python

from yaml import load
import argparse
import os

AUTHOR_TEMPLATE = '\\author[{}]{{{}}}\n'
ORGANIZATION_TEMPLATE = '\\affil[{}]{{{}}}\n'
TITLE_TEMPLATE = '\\title{{{}}}\n'

COVER_PAGE_TEMPLATE = """
\\newcommand{{\\makecoverpage}}{{
	\\begin{{titlepage}}
	\\center 
	\\textsc{{\\LARGE Illinois Data Science Initiative}} \\\\
	[1.5cm]\\textsc{{\\Large {}}} \\HRule \\\\
	[0.4cm]{{\\huge \\emph{{Version: }} {} }}\\\\
	\\HRule \\\\
	[1.5cm]\\Large \\emph{{Author(s): }} {} \\n \\\\
	[3.0cm] {{\\large \\today}} % Date
	%\\includegraphics{{Logo}}\\\\[1cm] % uncomment if you want to place a logo
	\\vfill
	\\end{{titlepage}}
}}
"""

CONFIG_FILE_NAME = 'config.yaml'
CONFIG_OUTPUT_NAME = 'config.cls'

VERSION_KEY = 'version'
REPORT_KEY_KEY = 'report_key'
REPORT_TITLE_KEY = 'report_title'
ORGANIZATIONS_KEY = 'organizations'
AUTHORS_KEY = 'authors'
PREREQS_KEY = 'prereqs'

def main(target_path):
	print("===============Config Creation==============")
	base = os.path.dirname(target_path)
	config_path = base + '/' + CONFIG_FILE_NAME
	output_path = base + '/' + CONFIG_OUTPUT_NAME
	try:
		os.stat(config_path)
	except Exception:
		print("Error: Config file not found for path {}".format(config_path))
		os.exit(1)

	config = load(open(config_path, "r"))
	output = "%%%%% THIS FILE WAS AUTOGENERATED %%%%%\n"
	for author, value in config[AUTHORS_KEY].items():
		value = list(map(str, value))
		output += AUTHOR_TEMPLATE.format(','.join(value), author)

	for i, item in enumerate(config[ORGANIZATIONS_KEY]):
		output += ORGANIZATION_TEMPLATE.format(str(i+1), item)
	output += TITLE_TEMPLATE.format(config[REPORT_TITLE_KEY])
	output += COVER_PAGE_TEMPLATE.format(config[REPORT_TITLE_KEY],
					config[VERSION_KEY],
				 ', '.join(config[AUTHORS_KEY].keys()))
	with open(output_path, "w+") as f:
		f.write(output)
	print("Successfully create config file for {}".format(config_path))
	print("============================================")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert some file')
	parser.add_argument('tex_path', type=str, help='A path to the tex file to convert')
	args = vars(parser.parse_args())
	main(args['tex_path'])