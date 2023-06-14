'''
Adapted from Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

from prettytable import PrettyTable

def eval_results(pixel_or_point,
                 settings,
                 recorder,
                 metrics_dict,
                 dataloader,
                 print_data_distribution=False,
                 print_confusion_matrix=False,
                 print_accuracy=False,
                 print_recall=False):


    # Metrics
    mean_acc, class_acc = metrics_dict['mean_acc'], metrics_dict['class_acc']
    mean_recall, class_recall = metrics_dict['mean_recall'], metrics_dict['class_recall']
    mean_iou, class_iou = metrics_dict['mean_iou'], metrics_dict['class_iou']


    dim = '3D' if pixel_or_point=='Point' else '2D'
    recorder.logger.info(f'============ {pixel_or_point}-wise Evaluation Results ({dim} Eval) ============')

    log_str = 'Acc avg: {:.4f}, IOU avg: {:.4f}, Recall avg: {:.4f}'.format(
        mean_acc.item(), mean_iou.item(), mean_recall.item())
    recorder.logger.info(log_str)

    cls_eval_table = PrettyTable(['Class ID', 'Class Name', 'IOU', 'Acc', 'Recall'])
    latext_str = ''
    for i, iou in enumerate(class_iou.cpu()):
        if i not in [0]:
            cls_eval_table.add_row([i, dataloader.dataset.mapped_cls_name[i], iou.item(), class_acc[i].cpu(
            ).item(), class_recall[i].cpu().item()])
            latext_str += ' & {:0.1f}'.format(iou * 100)
    latext_str += ' & {:0.1f}'.format(mean_iou.cpu().item() * 100)
    recorder.logger.info(cls_eval_table)
    recorder.logger.info('---- Latext Format String ----')
    recorder.logger.info(latext_str)

    # Confusion matrix
    conf_matrix = metrics_dict['conf_matrix']
    conf_matrix[0] = 0
    conf_matrix[:, 0] = 0

    if print_confusion_matrix:
        recorder.logger.info('---- Confusion Matrix Original Data ----')
        recorder.logger.info(conf_matrix)

    # Data distribution
    if print_data_distribution:
        distribution_table = PrettyTable(['Class Name', 'Number of Points', 'Percentage'])
        dist_data = conf_matrix.sum(0)
        for i in range(settings.n_classes):
            distribution_table.add_row(
                [dataloader.dataset.mapped_cls_name[i], dist_data[i].item(), str('{:.4f}'.format((dist_data[i]/dist_data.sum()).item() * 100)) + '%'])
        recorder.logger.info('---- Data Distribution ----')
        recorder.logger.info(distribution_table)

    # Accuracy metrics
    if print_accuracy:
        acc_data = conf_matrix.float() / (conf_matrix.sum(1, keepdim=True).float() + 1e-8)
        table_title = [' ']
        for i in range(1, settings.n_classes):
            table_title.append('{}'.format(
                dataloader.dataset.mapped_cls_name[i]))
        acc_table = PrettyTable(table_title)
        for i in range(1, settings.n_classes):
            row_data = ['{}'.format(
                dataloader.dataset.mapped_cls_name[i])]
            for j in range(1, settings.n_classes):
                row_data.append('{:0.1f}'.format(acc_data[i, j]*100))
            acc_table.add_row(row_data)
        recorder.logger.info('---- ACC Matrix ----')
        recorder.logger.info(acc_table)

    # Recall metrics
    if print_recall:
        recall_data = conf_matrix.float() / (conf_matrix.sum(0, keepdim=True).float()+1e-8)
        table_title = [' ']
        for i in range(1, settings.n_classes):
            table_title.append('{}'.format(
                dataloader.dataset.mapped_cls_name[i]))
        recall_table = PrettyTable(table_title)
        for i in range(1, settings.n_classes):
            row_data = ['{}'.format(
                dataloader.dataset.mapped_cls_name[i])]
            for j in range(1, settings.n_classes):
                row_data.append('{:0.1f}'.format(recall_data[i, j]*100))
            recall_table.add_row(row_data)
        recorder.logger.info('---- Recall Matrix ----')
        recorder.logger.info(recall_table)
