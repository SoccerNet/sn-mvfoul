
import os
import torch
import json
from config.classes import EVENT_DICTIONARY

##############################################################
#                                                            #
#              DO NOT MAKE ANY CHANGES HERE                  #
#                                                            #
##############################################################



# Function to load the labels from the json file
def label2vectormerge(folder_path, split, num_views):
	path_annotations = os.path.join(folder_path, split)
	path_annotations = os.path.join(path_annotations, "annotations.json") 

	dictionary_action = EVENT_DICTIONARY['action_class']

	if os.path.exists(path_annotations):
		with open(path_annotations) as f:
			train_annotations_data = json.load(f)
	else:
		print("PATH DOES NOT EXISTS")
		exit()

	not_taking = []

	num_classes_action = 8
	num_classes_offence_severity = 4

	labels_action = []
	labels_offence_severity= []
	number_of_actions = []

	total_distribution = torch.zeros(num_classes_offence_severity, num_classes_action)
	distribution_action = torch.zeros(1, num_classes_action)
	distribution_offence_severity = torch.zeros(1, num_classes_offence_severity)

	for actions in train_annotations_data['Actions']:
		action_class = train_annotations_data['Actions'][actions]['Action class']
		offence_class = train_annotations_data['Actions'][actions]['Offence']
		severity_class = train_annotations_data['Actions'][actions]['Severity']


		if action_class == '' or action_class == 'Dont know':
			not_taking.append(actions)
			continue

		if (offence_class == '' or offence_class == 'Between') and action_class != 'Dive':
			not_taking.append(actions)
			continue

		if (severity_class == '' or severity_class == '2.0' or severity_class == '4.0') and action_class != 'Dive' and offence_class != 'No offence' and offence_class != 'No Offence':
			not_taking.append(actions)
			continue

		if offence_class == '' or offence_class == 'Between':
			offence_class = 'Offence'

		if severity_class == '' or severity_class == '2.0' or severity_class == '4.0':
			severity_class = '1.0'

		if num_views == 1:
			for i in range(len(train_annotations_data['Actions'][actions]['Clips'])):
				if offence_class == 'No Offence' or offence_class == 'No offence':
					labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
					labels_offence_severity[len(labels_offence_severity)-1][0][0] = 1
					distribution_offence_severity[0][0] += 1
					off_index = 0
				elif offence_class == 'Offence' and severity_class == '1.0':
					labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
					labels_offence_severity[len(labels_offence_severity)-1][0][1] = 1
					distribution_offence_severity[0][1] += 1
					off_index = 1
				elif offence_class == 'Offence' and severity_class == '3.0':
					labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
					labels_offence_severity[len(labels_offence_severity)-1][0][2] = 1
					distribution_offence_severity[0][2] += 1
					off_index = 2
				elif offence_class == 'Offence' and severity_class == '5.0':
					labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
					labels_offence_severity[len(labels_offence_severity)-1][0][3] = 1
					distribution_offence_severity[0][3] += 1
					off_index = 3
				else:
					not_taking.append(actions)
					continue
				labels_action.append(torch.zeros(1, num_classes_action))
				labels_action[len(labels_action)-1][0][dictionary_action[action_class]] = 1
				distribution_action[0][dictionary_action[action_class]] += 1
				total_distribution[off_index][dictionary_action[action_class]] += 1
		else:
			if offence_class == 'No Offence' or offence_class == 'No offence':
				labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
				labels_offence_severity[len(labels_offence_severity)-1][0][0] = 1
				distribution_offence_severity[0][0] += 1
				index = 0
			elif offence_class == 'Offence' and severity_class == '1.0':
				labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
				labels_offence_severity[len(labels_offence_severity)-1][0][1] = 1
				distribution_offence_severity[0][1] += 1
				index = 1
			elif offence_class == 'Offence' and severity_class == '3.0':
				labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
				labels_offence_severity[len(labels_offence_severity)-1][0][2] = 1
				distribution_offence_severity[0][2] += 1
				index = 2
			elif offence_class == 'Offence' and severity_class == '5.0':
				labels_offence_severity.append(torch.zeros(1, num_classes_offence_severity))
				labels_offence_severity[len(labels_offence_severity)-1][0][3] = 1
				distribution_offence_severity[0][3] += 1
				index = 3
			else:
				not_taking.append(actions)
				continue

			number_of_actions.append(actions)
			labels_action.append(torch.zeros(1, num_classes_action))
			labels_action[len(labels_action)-1][0][dictionary_action[action_class]] = 1
			distribution_action[0][dictionary_action[action_class]] += 1
			total_distribution[index][dictionary_action[action_class]] += 1

	return labels_offence_severity, labels_action, distribution_offence_severity[0], distribution_action[0], not_taking, number_of_actions


# Function to load the path to the clips
def clips2vectormerge(folder_path, split, num_views, not_taking):

	path_clips = os.path.join(folder_path, split)

	if os.path.exists(path_clips):
		folders = 0

		for _, dirnames, _ in os.walk(path_clips):
			folders += len(dirnames) 
			
		clips = []
		for i in range(folders):
			if str(i) in not_taking:
				continue
			
			if num_views == 1:
				path_clip = os.path.join(path_clips, "action_" + str(i))
				path_clip_0 = os.path.join(path_clip, "clip_0.mp4")
				clips_all_view = []
				clips_all_view.append(path_clip_0)
				clips.append(clips_all_view)
				clips_all_view = []
				path_clip_1 = os.path.join(path_clip, "clip_1.mp4")
				clips_all_view.append(path_clip_1)
				clips.append(clips_all_view)
				clips_all_view = []

				if os.path.exists(os.path.join(path_clip, "clip_2.mp4")):
					path_clip_2 = os.path.join(path_clip, "clip_2.mp4")
					clips_all_view.append(path_clip_2)
					clips.append(clips_all_view)
					clips_all_view = []

				if os.path.exists(os.path.join(path_clip, "clip_3.mp4")):
					path_clip_3 = os.path.join(path_clip, "clip_3.mp4")
					clips_all_view.append(path_clip_3)
					clips.append(clips_all_view)
					clips_all_view = []
			else:
				path_clip = os.path.join(path_clips, "action_" + str(i))
				path_clip_0 = os.path.join(path_clip, "clip_0.mp4")
				clips_all_view = []
				clips_all_view.append(path_clip_0)
				path_clip_1 = os.path.join(path_clip, "clip_1.mp4")
				clips_all_view.append(path_clip_1)

				if os.path.exists(os.path.join(path_clip, "clip_2.mp4")):
					path_clip_2 = os.path.join(path_clip, "clip_2.mp4")
					clips_all_view.append(path_clip_2)

				if os.path.exists(os.path.join(path_clip, "clip_3.mp4")):
					path_clip_3 = os.path.join(path_clip, "clip_3.mp4")
					clips_all_view.append(path_clip_3)
				clips.append(clips_all_view)

		return clips


