import sys
import csv

with open('/mnt/volume/stackOverflowFormatted/data.csv','wb') as csvfile:
	writer = csv.writer(csvfile, delimiter='\t')
	writer.writerow(["Id","PostTypeId","AcceptedAnswerId","CreationDate","Score","ViewCount","Body","OwnerUserId","LastEditorUserId","LastEditorDisplayName","LastEditDate","LastActivityDate","Title","Tags","AnswerCount","CommentCount","FavoriteCount","CommunityOwnedDate"])

	for line in sys.stdin:
		if len(line)>50:
			split_line = line.split("\"")
			
			#Check if the value is a question
			if split_line[3] =="1": 
				result = []
				for i in xrange(1,36,2):
					if i<len(split_line):
						result.append(split_line[i])
					else:
						result.append("")
						
				#write to csv
				writer.writerow(res)