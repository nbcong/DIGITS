# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import tempfile
import operator

import flask
import werkzeug.exceptions

from .forms import GenericImageModelForm
from .job import GenericImageModelJob
from digits import frameworks
from digits import utils
from digits.config import config_value
from digits.dataset import GenericImageDatasetJob
from digits.inference import ImageInferenceJob
from digits.status import Status
from digits.utils import filesystem as fs
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np

from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler

blueprint = flask.Blueprint(__name__, __name__)

#paths to test model
psy_test_model = {
    'trained_model_path': '',
    'deploy_model_proto': '',
    'network': None,
    'net': None,
    'height': 180,
    'width': 320
}

#======================================================================================
def get_layer_statistics(data):
    """
    Returns statistics for the given layer data:
        (mean, standard deviation, histogram)
            histogram -- [y, x, ticks]

    Arguments:
    data -- a np.ndarray
    """
    # XXX These calculations can be super slow
    mean = np.mean(data)
    std = np.std(data)
    y, x = np.histogram(data, bins=20)
    y = list(y)
    ticks = x[[0,len(x)/2,-1]]
    x = [(x[i]+x[i+1])/2.0 for i in xrange(len(x)-1)]
    ticks = list(ticks)
    return (mean, std, [y, x, ticks])

def infer_one_generic(image, layers=None):
    """
    Run inference on one image for a generic model
    Returns (output, visualizations)
        output -- an dict of string -> np.ndarray
        visualizations -- a list of dicts for the specified layers
    Returns (None, None) if something goes wrong

    Arguments:
    image -- an np.ndarray

    Keyword arguments:
    snapshot_epoch -- which snapshot to use
    layers -- which layer activation[s] and weight[s] to visualize
    """

    net = None
    if psy_test_model['net'] == None and \
       psy_test_model['deploy_model_proto'] != '' and \
       psy_test_model['trained_model_path'] != '':
        psy_test_model['net'] = caffe.Net(psy_test_model['deploy_model_proto'], psy_test_model['trained_model_path'], caffe.TEST)
        network = caffe_pb2.NetParameter()
        with open(psy_test_model['deploy_model_proto']) as infile:
            text_format.Merge(infile.read(), network)
        psy_test_model['network'] = network
        # TODO:
        # we can get the width, height from network
        # network.input_shape
        # will make a change later
        
    net = psy_test_model['net']

    image = np.array(image, dtype=np.float32)

    # TODO: hacky
    mean = np.array((104.00698793,116.66876762,122.67891434))
    # image is formatted as RGB, convert to BGR
    image = image[:, :, ::-1]
    image -= mean
    image = image.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *image.shape)
    net.blobs['data'].data[...] = image
    
    output = net.forward()
    
    visualizations = get_layer_visualizations(net, layers)
    
    return (output, visualizations)

def get_layer_visualizations(net, layers='all'):
    """
    Returns visualizations of various layers in the network
    """
    network = psy_test_model['network']
    # add visualizations
    visualizations = []
    if layers and layers != 'none':
        if layers == 'all':
            added_activations = []
            for layer in network.layer:
                print 'Computing visualizations for "%s" ...' % layer.name
                for bottom in layer.bottom:
                    if bottom in net.blobs and bottom not in added_activations:
                        data = net.blobs[bottom].data[0]
                        vis = utils.image.get_layer_vis_square(data,
                                allow_heatmap=bool(bottom != 'data'))
                        mean, std, hist = get_layer_statistics(data)
                        visualizations.append(
                                {
                                    'name': str(bottom),
                                    'vis_type': 'Activation',
                                    'image_html': utils.image.embed_image_html(vis),
                                    'data_stats': {
                                        'shape': data.shape,
                                        'mean': mean,
                                        'stddev': std,
                                        'histogram': hist,
                                        },
                                    }
                                )
                        added_activations.append(bottom)
                if layer.name in net.params:
                    data = net.params[layer.name][0].data
                    if layer.type not in ['InnerProduct']:
                        vis = utils.image.get_layer_vis_square(data)
                    else:
                        vis = None
                    mean, std, hist = get_layer_statistics(data)
                    params = net.params[layer.name]
                    weight_count = reduce(operator.mul, params[0].data.shape, 1)
                    if len(params) > 1:
                        bias_count = reduce(operator.mul, params[1].data.shape, 1)
                    else:
                        bias_count = 0
                    parameter_count = weight_count + bias_count
                    visualizations.append(
                            {
                                'name': str(layer.name),
                                'vis_type': 'Weights',
                                'layer_type': layer.type,
                                'param_count': parameter_count,
                                'image_html': utils.image.embed_image_html(vis),
                                'data_stats': {
                                    'shape':data.shape,
                                    'mean': mean,
                                    'stddev': std,
                                    'histogram': hist,
                                    },
                                }
                            )
                for top in layer.top:
                    if top in net.blobs and top not in added_activations:
                        data = net.blobs[top].data[0]
                        normalize = True
                        # don't normalize softmax layers
                        if layer.type == 'Softmax':
                            normalize = False
                        vis = utils.image.get_layer_vis_square(data,
                                normalize = normalize,
                                allow_heatmap = bool(top != 'data'))
                        mean, std, hist = get_layer_statistics(data)
                        visualizations.append(
                                {
                                    'name': str(top),
                                    'vis_type': 'Activation',
                                    'image_html': utils.image.embed_image_html(vis),
                                    'data_stats': {
                                        'shape': data.shape,
                                        'mean': mean,
                                        'stddev': std,
                                        'histogram': hist,
                                        },
                                    }
                                )
                        added_activations.append(top)
        else:
            raise NotImplementedError

    return visualizations
#======================================================================================













@blueprint.route('/new', methods=['GET'])
@utils.auth.requires_login
def new():
    """
    Return a form for a new GenericImageModelJob
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('models/images/generic/new.html',
            form = form,
            frameworks = frameworks.get_frameworks(),
            previous_network_snapshots = prev_network_snapshots,
            previous_networks_fullinfo = get_previous_networks_fulldetails(),
            multi_gpu = config_value('caffe_root')['multi_gpu'],
            )

@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Create a new GenericImageModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/generic/new.html',
                    form = form,
                    frameworks = frameworks.get_frameworks(),
                    previous_network_snapshots = prev_network_snapshots,
                    previous_networks_fullinfo = get_previous_networks_fulldetails(),
                    multi_gpu = config_value('caffe_root')['multi_gpu'],
                    ), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
                'Unknown dataset job_id "%s"' % form.dataset.data)

    # sweeps will be a list of the the permutations of swept fields
    # Get swept learning_rate
    sweeps = [{'learning_rate': v} for v in form.learning_rate.data]
    add_learning_rate = len(form.learning_rate.data) > 1

    # Add swept batch_size
    sweeps = [dict(s.items() + [('batch_size', bs)]) for bs in form.batch_size.data for s in sweeps[:]]
    add_batch_size = len(form.batch_size.data) > 1
    n_jobs = len(sweeps)

    jobs = []
    for sweep in sweeps:
        # Populate the form with swept data to be used in saving and
        # launching jobs.
        form.learning_rate.data = sweep['learning_rate']
        form.batch_size.data = sweep['batch_size']

        # Augment Job Name
        extra = ''
        if add_learning_rate:
            extra += ' learning_rate:%s' % str(form.learning_rate.data[0])
        if add_batch_size:
            extra += ' batch_size:%d' % form.batch_size.data[0]

        job = None
        try:
            job = GenericImageModelJob(
                    username    = utils.auth.get_username(),
                    name        = form.model_name.data + extra,
                    dataset_id  = datasetJob.id(),
                    )

            # get framework (hard-coded to caffe for now)
            fw = frameworks.get_framework_by_id(form.framework.data)

            pretrained_model = None
            #if form.method.data == 'standard':
            if form.method.data == 'previous':
                old_job = scheduler.get_job(form.previous_networks.data)
                if not old_job:
                    raise werkzeug.exceptions.BadRequest(
                            'Job not found: %s' % form.previous_networks.data)

                use_same_dataset = (old_job.dataset_id == job.dataset_id)
                network = fw.get_network_from_previous(old_job.train_task().network, use_same_dataset)

                for choice in form.previous_networks.choices:
                    if choice[0] == form.previous_networks.data:
                        epoch = float(flask.request.form['%s-snapshot' % form.previous_networks.data])
                        if epoch == 0:
                            pass
                        elif epoch == -1:
                            pretrained_model = old_job.train_task().pretrained_model
                        else:
                            for filename, e in old_job.train_task().snapshots:
                                if e == epoch:
                                    pretrained_model = filename
                                    break

                            if pretrained_model is None:
                                raise werkzeug.exceptions.BadRequest(
                                        "For the job %s, selected pretrained_model for epoch %d is invalid!"
                                        % (form.previous_networks.data, epoch))
                            if not (os.path.exists(pretrained_model)):
                                raise werkzeug.exceptions.BadRequest(
                                        "Pretrained_model for the selected epoch doesn't exists. May be deleted by another user/process. Please restart the server to load the correct pretrained_model details")
                        break

            elif form.method.data == 'custom':
                network = fw.get_network_from_desc(form.custom_network.data)
                pretrained_model = form.custom_network_snapshot.data.strip()
            else:
                raise werkzeug.exceptions.BadRequest(
                        'Unrecognized method: "%s"' % form.method.data)

            policy = {'policy': form.lr_policy.data}
            if form.lr_policy.data == 'fixed':
                pass
            elif form.lr_policy.data == 'step':
                policy['stepsize'] = form.lr_step_size.data
                policy['gamma'] = form.lr_step_gamma.data
            elif form.lr_policy.data == 'multistep':
                policy['stepvalue'] = form.lr_multistep_values.data
                policy['gamma'] = form.lr_multistep_gamma.data
            elif form.lr_policy.data == 'exp':
                policy['gamma'] = form.lr_exp_gamma.data
            elif form.lr_policy.data == 'inv':
                policy['gamma'] = form.lr_inv_gamma.data
                policy['power'] = form.lr_inv_power.data
            elif form.lr_policy.data == 'poly':
                policy['power'] = form.lr_poly_power.data
            elif form.lr_policy.data == 'sigmoid':
                policy['stepsize'] = form.lr_sigmoid_step.data
                policy['gamma'] = form.lr_sigmoid_gamma.data
            else:
                raise werkzeug.exceptions.BadRequest(
                        'Invalid learning rate policy')

            if config_value('caffe_root')['multi_gpu']:
                if form.select_gpu_count.data:
                    gpu_count = form.select_gpu_count.data
                    selected_gpus = None
                else:
                    selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                    gpu_count = None
            else:
                if form.select_gpu.data == 'next':
                    gpu_count = 1
                    selected_gpus = None
                else:
                    selected_gpus = [str(form.select_gpu.data)]
                    gpu_count = None

            # Python Layer File may be on the server or copied from the client.
            fs.copy_python_layer_file(
                bool(form.python_layer_from_client.data),
                job.dir(),
                (flask.request.files[form.python_layer_client_file.name]
                 if form.python_layer_client_file.name in flask.request.files
                 else ''), form.python_layer_server_file.data)

            job.tasks.append(fw.create_train_task(
                        job_dir         = job.dir(),
                        dataset         = datasetJob,
                        train_epochs    = form.train_epochs.data,
                        snapshot_interval   = form.snapshot_interval.data,
                        learning_rate   = form.learning_rate.data[0],
                        lr_policy       = policy,
                        gpu_count       = gpu_count,
                        selected_gpus   = selected_gpus,
                        batch_size      = form.batch_size.data[0],
                        val_interval    = form.val_interval.data,
                        pretrained_model= pretrained_model,
                        crop_size       = form.crop_size.data,
                        use_mean        = form.use_mean.data,
                        network         = network,
                        random_seed     = form.random_seed.data,
                        solver_type     = form.solver_type.data,
                        shuffle         = form.shuffle.data,
                        )
                    )

            ## Save form data with the job so we can easily clone it later.
            save_form_to_job(job, form)

            jobs.append(job)
            scheduler.add_job(job)
            if n_jobs == 1:
                if request_wants_json():
                    return flask.jsonify(job.json_dict())
                else:
                    return flask.redirect(flask.url_for('digits.model.views.show', job_id=job.id()))

        except:
            if job:
                scheduler.delete_job(job)
            raise

    if request_wants_json():
        return flask.jsonify(jobs=[job.json_dict() for job in jobs])

    # If there are multiple jobs launched, go to the home page.
    return flask.redirect('/')

def show(job):
    """
    Called from digits.model.views.models_show()
    """
    return flask.render_template('models/images/generic/show.html', job=job)

@blueprint.route('/large_graph', methods=['GET'])
def large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/generic/large_graph.html', job=job)

@blueprint.route('/infer_one.json', methods=['POST'])
@blueprint.route('/infer_one', methods=['POST', 'GET'])
def infer_one():
    """
    Infer one image
    """
    model_job = job_from_request()

    remove_image_path = False
    if 'image_path' in flask.request.form and flask.request.form['image_path']:
        image_path = flask.request.form['image_path']
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        outfile = tempfile.mkstemp(suffix='.bin')
        flask.request.files['image_file'].save(outfile[1])
        image_path = outfile[1]
        os.close(outfile[0])
    else:
        raise werkzeug.exceptions.BadRequest('must provide image_path or image_file')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    # create inference job
    inference_job = ImageInferenceJob(
                username    = utils.auth.get_username(),
                name        = "Infer One Image",
                model       = model_job,
                images      = [image_path],
                epoch       = epoch,
                layers      = layers
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, visualizations = inference_job.get_data()

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if remove_image_path:
        os.remove(image_path)

    image = None
    if inputs is not None and len(inputs['data']) == 1:
        image = utils.image.embed_image_html(inputs['data'][0])

    if request_wants_json():
        return flask.jsonify({'outputs': dict((name, blob.tolist()) for name,blob in outputs.iteritems())})
    else:
        return flask.render_template('models/images/generic/infer_one.html',
                model_job       = model_job,
                job             = inference_job,
                image_src       = image,
                network_outputs = outputs,
                visualizations  = visualizations,
                total_parameters= sum(v['param_count'] for v in visualizations if v['vis_type'] == 'Weights'),
                )

@blueprint.route('/psy_infer_one.json', methods=['POST'])
@blueprint.route('/psy_infer_one', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def generic_image_model_psy_infer_one():
    """
    Infer one image
    """

    image = None
    if 'psy_test_image_file' in flask.request.files and flask.request.files['psy_test_image_file']:
        outfile = tempfile.mkstemp(suffix='.bin')
        flask.request.files['psy_test_image_file'].save(outfile[1])
        image = utils.image.load_image(outfile[1])
        os.close(outfile[0])
        os.remove(outfile[1])
    else:
        raise werkzeug.exceptions.BadRequest('must provide test image file')

    psy_trained_model = None
    if 'psy_trained_network' in flask.request.form:
        psy_trained_model = str(flask.request.form['psy_trained_network'])
    if psy_trained_model == None or psy_trained_model == '':
        raise werkzeug.exceptions.BadRequest('must provide trained network')

    psy_deploy_model_proto = None
    if 'psy_deploy_network_proto' in flask.request.form:
        psy_deploy_model_proto = str(flask.request.form['psy_deploy_network_proto'])
    if psy_deploy_model_proto == None or psy_deploy_model_proto == '':
        raise werkzeug.exceptions.BadRequest('must provide deploy network proto')

    if psy_trained_model != psy_test_model['trained_model_path'] or \
        psy_deploy_model_proto != psy_test_model['deploy_model_proto']:
        psy_test_model['net'] = None
        psy_test_model['network'] = None

    psy_test_model['trained_model_path'] = psy_trained_model
    psy_test_model['deploy_model_proto'] = psy_deploy_model_proto
    
    # resize image
    auto_resize_image = True
    if auto_resize_image:
        height = psy_test_model['height']
        width = psy_test_model['width']
        image = utils.image.resize_image(image, height, width,
                channels = 3,
                resize_mode = 'squash',
                )

    # show visualizations
    layers = 'all'
    
    outputs, visualizations = infer_one_generic(image, layers=layers)

    return flask.render_template('models/images/generic/infer_one.html',
                job             = None,
                image_src       = utils.image.embed_image_html(image),
                network_outputs = outputs,
                visualizations  = visualizations,
                total_parameters= sum(v['param_count'] for v in visualizations if v['vis_type'] == 'Weights'),
                )
    

@blueprint.route('/infer_many.json', methods=['POST'])
@blueprint.route('/infer_many', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def infer_many():
    """
    Infer many images
    """
    model_job = job_from_request()

    image_list = flask.request.files.get('image_list')
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    if 'image_folder' in flask.request.form and flask.request.form['image_folder'].strip():
        image_folder = flask.request.form['image_folder']
        if not os.path.exists(image_folder):
            raise werkzeug.exceptions.BadRequest('image_folder "%s" does not exit' % image_folder)
    else:
        image_folder = None

    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_test_images = int(flask.request.form['num_test_images'])
    else:
        num_test_images = None

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths = []

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+\d+$', line)
        if match:
            path = match.group(1)
        else:
            path = line

        if not utils.is_url(path) and image_folder and not os.path.isabs(path):
            path = os.path.join(image_folder, path)
        paths.append(path)

        if num_test_images is not None and len(paths) >= num_test_images:
            break

    # create inference job
    inference_job = ImageInferenceJob(
                username    = utils.auth.get_username(),
                name        = "Infer Many Images",
                model       = model_job,
                images      = paths,
                epoch       = epoch,
                layers      = 'none'
                )

    # schedule tasks
    scheduler.add_job(inference_job)

    # wait for job to complete
    inference_job.wait_completion()

    # retrieve inference data
    inputs, outputs, _ = inference_job.get_data()

    # delete job folder and remove from scheduler list
    scheduler.delete_job(inference_job)

    if outputs is not None and len(outputs) < 1:
        # an error occurred
        outputs = None

    if inputs is not None:
        paths = [paths[idx] for idx in inputs['ids']]

    if request_wants_json():
        result = {}
        for i, path in enumerate(paths):
            result[path] = dict((name, blob[i].tolist()) for name,blob in outputs.iteritems())
        return flask.jsonify({'outputs': result})
    else:
        return flask.render_template('models/images/generic/infer_many.html',
                model_job       = model_job,
                job             = inference_job,
                paths           = paths,
                network_outputs = outputs,
                )

def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, GenericImageDatasetJob) and (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, GenericImageModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, GenericImageModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_network_snapshots():
    prev_network_snapshots = []
    for job_id, _ in get_previous_networks():
        job = scheduler.get_job(job_id)
        e = [(0, 'None')] + [(epoch, 'Epoch #%s' % epoch)
                for _, epoch in reversed(job.train_task().snapshots)]
        if job.train_task().pretrained_model:
            e.insert(0, (-1, 'Previous pretrained model'))
        prev_network_snapshots.append(e)
    return prev_network_snapshots

