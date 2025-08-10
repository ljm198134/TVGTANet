from core import runner
import torch
import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Textual and Visual Guided Task Adaptation for Source-free Cross-Domain Few-Shot Segmentation')
    parser.add_argument('--benchmark', type=str, default='fss',choices=['fss','deepglobe','lung','isic'])
    parser.add_argument('--datapath', type=str, default=r"D:\\CDFSSDataSet\\Target domains")
    parser.add_argument('--nshot', type=int, default=1,choices=[1,5])
    parser.add_argument('--adapt-to', type=str, default='first-episode', choices=['first-episode', 'every-episode'])
    parser.add_argument('--postprocessing', type=str, default='dynamic', choices=['off', 'dynamic', 'always'])
    parser.add_argument('--logpath', type=str, default='./logs')
    parser.add_argument('--verbosity', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_opts()
    runner.set_args(args)
    dataloader = runner.makeDataloader()
    config = runner.makeConfig()
    config.featext.fit_every_episode = args.adapt_to == 'every-episode'
    feat_maker = runner.makeFeatureMaker(dataloader.dataset, config, device=device)
    average_meter = runner.AverageMeterWrapper(dataloader, device)
    for idx, batch in enumerate(dataloader):
        sseval = runner.SingleSampleEval(batch, feat_maker)
        sseval.post_proc_method = args.postprocessing
        sseval.forward()
        sseval.calc_metrics()
        average_meter.update(sseval)
        average_meter.write(idx)
    print('Result m|FB:', average_meter.average_meter.compute_iou())
