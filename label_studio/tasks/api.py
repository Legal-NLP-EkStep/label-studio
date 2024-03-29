"""This file and its contents are licensed under the Apache License 2.0. Please see the included NOTICE for copyright information and LICENSE for a copy of the license.
"""
import logging

from django.db.models import Q
from django.utils import timezone

import drf_yasg.openapi as openapi
from drf_yasg.utils import swagger_auto_schema

from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from core.utils.common import get_object_with_check_and_log
from core.decorators import permission_required
from core.permissions import all_permissions, ViewClassPermission

from tasks.models import Task, Annotation, Prediction, AnnotationDraft
from core.permissions import (get_object_with_permissions, check_object_permissions)
from core.mixins import RequestDebugLogMixin
from core.utils.common import bool_from_request
from tasks.serializers import (
    TaskSerializer, AnnotationSerializer, TaskSimpleSerializer, PredictionSerializer,
    TaskWithAnnotationsAndPredictionsAndDraftsSerializer, AnnotationDraftSerializer)
from projects.models import Project

logger = logging.getLogger(__name__)


class TaskListAPI(generics.ListCreateAPIView):
    """
    post:
    Create task

    Create a new labeling task in Label Studio.
    """
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

    @swagger_auto_schema(auto_schema=None)
    def get(self, request, *args, **kwargs):
        return super(TaskListAPI, self).get(request, *args, **kwargs)

    def get_serializer_context(self):
        context = super(TaskListAPI, self).get_serializer_context()
        project_id = self.request.data.get('project')
        if project_id:
            context['project'] = generics.get_object_or_404(Project, pk=project_id)
        return context

    @swagger_auto_schema(tags=['Tasks'], request_body=TaskSerializer)
    def post(self, request, *args, **kwargs):
        return super(TaskListAPI, self).post(request, *args, **kwargs)


class TaskAPI(generics.RetrieveUpdateDestroyAPIView):
    """
    get:
    Get task by ID

    Get task data, metadata, annotations and other attributes for a specific labeling task.

    patch:
    Update task

    Update the attributes of an existing labeling task.

    delete:
    Delete task

    Delete a task in Label Studio. This action cannot be undone!
    """
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    queryset = Task.objects.all()

    def get_serializer_class(self):
        # GET => task + annotations + predictions + drafts
        if self.request.method == 'GET':
            return TaskWithAnnotationsAndPredictionsAndDraftsSerializer

        # POST, PATCH, PUT
        else:
            return TaskSimpleSerializer

    def retrieve(self, request, *args, **kwargs):
        task = self.get_object()

        # call machine learning api and format response
        if task.project.evaluate_predictions_automatically:
            for ml_backend in task.project.ml_backends.all():
                ml_backend.predict_one_task(task)

        result = self.get_serializer(task).data

        # use proxy inlining to task data (for credential access)
        proxy = bool_from_request(request.GET, 'proxy', True)
        result['data'] = task.resolve_uri(result['data'], proxy=proxy)
        return Response(result)

    @swagger_auto_schema(tags=['Tasks'], manual_parameters=[
        openapi.Parameter(name='proxy', type=openapi.TYPE_BOOLEAN, in_=openapi.IN_QUERY,
                          description='Use the proxy parameter inline for credential access to task data')
    ])
    def get(self, request, *args, **kwargs):
        return super(TaskAPI, self).get(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Tasks'], request_body=TaskSimpleSerializer)
    def patch(self, request, *args, **kwargs):
        return super(TaskAPI, self).patch(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Tasks'])
    def delete(self, request, *args, **kwargs):
        return super(TaskAPI, self).delete(request, *args, **kwargs)

    @swagger_auto_schema(auto_schema=None)
    def put(self, request, *args, **kwargs):
        return super(TaskAPI, self).put(request, *args, **kwargs)


class AnnotationAPI(RequestDebugLogMixin, generics.RetrieveUpdateDestroyAPIView):
    """
    get:
    Get annotation by its ID

    Retrieve a specific annotation for a task.

    patch:
    Update annotation

    Update existing attributes on an annotation. 

    delete:
    Delete annotation

    Delete an annotation. This action can't be undone! 
    """
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        PUT=all_permissions.annotations_change,
        PATCH=all_permissions.annotations_change,
        DELETE=all_permissions.annotations_delete,
    )

    serializer_class = AnnotationSerializer
    queryset = Annotation.objects.all()

    def perform_destroy(self, annotation):
        annotation.delete()

    @staticmethod
    # def find_and_annotate_duplicates(request, Annotation, annotation_id):
    #     import re
    #     import uuid
    #     import copy
    #     from tqdm import tqdm
    #     inclusion_list = ['COURT', 'AUTHORITY', 'POLICE_STATION', 'COMPANY', 'OTHER_ORG', 'STATUTE', 'PROVISION',
    #                       'PRECEDENT', 'LC_CASE', 'Appellant_Name', 'Respondent_Name', 'Judge_Name', 'Lawyer_Name',
    #                       'Witness_Name', 'Investigating_Officer', 'Other_Name', 'Accused_Name', 'Victim_Name',
    #                       'Convicted_Name', 'Acquitted_Name','Liable_Name','Not_Liable_Name','Petitioner_Name']
    #
    #     def struct(start, end, label, text):
    #         uid = uuid.uuid4()
    #         id = uid.hex
    #         data = {"value": {"start": start, "end": end, "text": text, "labels": [label]}, "id": id,
    #                 "from_name": "label",
    #                 "to_name": "text", "type": "labels"}
    #         return data
    #
    #     def de_duplicate_on_enity_label(old_annotations):
    #         complete_annotations = {}
    #         final_annotations = []
    #         for old_annotation in old_annotations:
    #             if old_annotation['value']['labels'][0] not in complete_annotations.keys():
    #                 complete_annotations[old_annotation['value']['labels'][0]] = []
    #
    #             complete_annotations[old_annotation['value']['labels'][0]].append(old_annotation)
    #
    #         for ann in complete_annotations.keys():
    #             annotations = complete_annotations[ann]
    #             annotations_sorted = sorted(annotations, key=lambda i: (i['value']['start'], -i['value']['end']))
    #             to_compare = annotations_sorted[0]
    #             annotations_sorted = annotations_sorted[1:]
    #
    #             final_annotations.append(to_compare)
    #             i = 0
    #             while i < len(annotations_sorted):
    #                 if annotations_sorted[i]['value']['end'] > to_compare['value']['end']:
    #                     to_compare = annotations_sorted[i]
    #                     final_annotations.append(to_compare)
    #
    #                 i += 1
    #         return final_annotations
    #
    #     def de_duplicate(list1, key=False):
    #         dummy_val = 0
    #         if key:
    #             check = {}
    #             list2 = []
    #             for i in list1:
    #                 key = i['value']['text'] + '@' + i['value']['labels'][0]
    #                 key = key.strip()
    #                 try:
    #                     val = check[key]
    #                 except:
    #                     check[key] = dummy_val
    #                     list2.append(i)
    #         else:
    #             list2 = []
    #             for i in list1:
    #                 if i not in list2:
    #                     list2.append(i)
    #         return list2
    #
    #     def de_duplicate_on_span_and_type(list1):
    #         dummy_val = 0
    #         check = {}
    #         list2 = []
    #         for i in list1:
    #             key = '@'.join(
    #                 [str(i['value']['start']), str(i['value']['end']), i['value']['text'], i['value']['labels'][0]])
    #             key = key.strip()
    #             try:
    #                 val = check[key]
    #             except:
    #                 check[key] = dummy_val
    #                 list2.append(i)
    #         return list2
    #
    #     def fetch_diff(list1, list2):
    #         list3 = []
    #         for i in list1:
    #             if not i in list2:
    #                 list3.append(i)
    #         return list3
    #
    #     obj = get_object_with_check_and_log(request, Annotation, pk=annotation_id)
    #     last_update = copy.deepcopy(obj.result)
    #     text = copy.deepcopy(obj.task.data['text'])
    #     draft = copy.deepcopy(request.data['result'])
    #     deleted = []  # fetch_diff(last_update, draft)
    #     added = fetch_diff(draft, last_update)
    #     new_draft = []
    #     for deleted_annotation in deleted:
    #         for i in draft:
    #             if not '@'.join(
    #                     [deleted_annotation['value']['text'],
    #                      deleted_annotation['value']['labels'][0]]).strip() == '@'.join(
    #                     [i['value']['text'], i['value']['labels'][0]]).strip():
    #                 new_draft.append(i)
    #     if not new_draft and not deleted:
    #         new_draft = draft
    #     new_added = de_duplicate(added, key=True)
    #     change = fetch_diff(added, new_added)
    #     for i in change:
    #         new_draft.remove(i)
    #     added = new_added
    #     for annotation in tqdm(added):
    #         annotated_text = annotation['value']['text']
    #         label = annotation['value']['labels'][0]  # ToDo Add inclusion list
    #         original_start = annotation['value']['start']
    #         original_end = annotation['value']['end']
    #         pattern = re.compile(r'\b' + '\s+'.join([re.escape(i) for i in annotated_text.split()]) + '(?=\W)',re.IGNORECASE)
    #         if label in inclusion_list:
    #             for found in pattern.finditer(text):
    #                 start = found.span()[0]
    #                 end = found.span()[1]
    #                 if start != original_start and end != original_end:
    #                     new_draft.append(struct(start, end, label, annotated_text))
    #     request.data['result'] = de_duplicate_on_enity_label(de_duplicate_on_span_and_type(de_duplicate(new_draft)))
    #     return request

    def update(self, request, *args, **kwargs):
        # save user history with annotator_id, time & annotation result
        annotation_id = self.kwargs['pk']
        # request = self.find_and_annotate_duplicates(request, Annotation, annotation_id)
        annotation = get_object_with_check_and_log(request, Annotation, pk=annotation_id)
        annotation.task.save()  # refresh task metrics

        return super(AnnotationAPI, self).update(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Annotations'])
    def get(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).get(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Annotations'], auto_schema=None)
    def put(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).put(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Annotations'], request_body=AnnotationSerializer)
    def patch(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).patch(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Annotations'])
    def delete(self, request, *args, **kwargs):
        return super(AnnotationAPI, self).delete(request, *args, **kwargs)


class AnnotationsListAPI(RequestDebugLogMixin, generics.ListCreateAPIView):
    """
    get:
    Get all task annotations

    List all annotations for a task.

    post:
    Create new annotation

    Add annotations to a task like an annotator does.
    """
    parser_classes = (JSONParser, FormParser, MultiPartParser)
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        POST=all_permissions.annotations_create,
    )

    serializer_class = AnnotationSerializer

    @swagger_auto_schema(tags=['Annotations'])
    def get(self, request, *args, **kwargs):
        return super(AnnotationsListAPI, self).get(request, *args, **kwargs)

    @swagger_auto_schema(tags=['Annotations'], request_body=AnnotationSerializer)
    def post(self, request, *args, **kwargs):
        return super(AnnotationsListAPI, self).post(request, *args, **kwargs)

    def get_queryset(self):
        task = generics.get_object_or_404(Task.objects.for_user(self.request.user), pk=self.kwargs.get('pk', 0))
        return Annotation.objects.filter(Q(task=task) & Q(was_cancelled=False)).order_by('pk')

    def perform_create(self, ser):
        task = get_object_with_check_and_log(self.request, Task, pk=self.kwargs['pk'])
        # annotator has write access only to annotations and it can't be checked it after serializer.save()
        user = self.request.user
        # Release task if it has been taken at work (it should be taken by the same user, or it makes sentry error
        logger.debug(f'User={user} releases task={task}')
        task.release_lock(user)

        # updates history
        update_id = self.request.user.id
        result = ser.validated_data.get('result')
        extra_args = {'task_id': self.kwargs['pk']}

        # save stats about how well annotator annotations coincide with current prediction
        # only for finished task annotations
        if result is not None:
            prediction = Prediction.objects.filter(task=task, model_version=task.project.model_version)
            if prediction.exists():
                prediction = prediction.first()
                prediction_ser = PredictionSerializer(prediction).data
            else:
                logger.debug(f'User={self.request.user}: there are no predictions for task={task}')
                prediction_ser = {}
            # serialize annotation
            extra_args.update({
                'prediction': prediction_ser,
            })

        if 'was_cancelled' in self.request.GET:
            extra_args['was_cancelled'] = bool_from_request(self.request.GET, 'was_cancelled', False)

        if 'completed_by' not in ser.validated_data:
            extra_args['completed_by'] = self.request.user

        # create annotation
        logger.debug(f'User={self.request.user}: save annotation')
        annotation = ser.save(**extra_args)
        logger.debug(f'Save activity for user={self.request.user}')
        self.request.user.activity_at = timezone.now()
        self.request.user.save()

        # if annotation created from draft - remove this draft
        draft_id = self.request.data.get('draft_id')
        if draft_id is not None:
            logger.debug(f'Remove draft {draft_id} after creating annotation {annotation.id}')
            AnnotationDraft.objects.filter(id=draft_id).delete()

        return annotation


class AnnotationDraftListAPI(RequestDebugLogMixin, generics.ListCreateAPIView):
    parser_classes = (JSONParser, MultiPartParser, FormParser)
    serializer_class = AnnotationDraftSerializer
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        POST=all_permissions.annotations_create,
    )
    queryset = AnnotationDraft.objects.all()
    swagger_schema = None

    def filter_queryset(self, queryset):
        task_id = self.kwargs['pk']
        return queryset.filter(task_id=task_id)

    def perform_create(self, serializer):
        task_id = self.kwargs['pk']
        annotation_id = self.kwargs.get('annotation_id')
        user = self.request.user
        logger.debug(f'User {user} is going to create draft for task={task_id}, annotation={annotation_id}')
        serializer.save(
            task_id=self.kwargs['pk'],
            annotation_id=annotation_id,
            user=self.request.user
        )


class AnnotationDraftAPI(RequestDebugLogMixin, generics.RetrieveUpdateDestroyAPIView):
    parser_classes = (JSONParser, MultiPartParser, FormParser)
    serializer_class = AnnotationDraftSerializer
    queryset = AnnotationDraft.objects.all()
    permission_required = ViewClassPermission(
        GET=all_permissions.annotations_view,
        PUT=all_permissions.annotations_change,
        PATCH=all_permissions.annotations_change,
        DELETE=all_permissions.annotations_delete,
    )
    swagger_schema = None
