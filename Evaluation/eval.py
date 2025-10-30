import os
import torch
import pickle



def main():
    info = torch.load('../log_nas/checkpoint.pth.tar')['vis_dict']
    cands = sorted([cand for cand in info if 'miou_with_dice' in info[cand]],
                   key=lambda cand: info[cand]['miou_with_dice'],reverse=True)[:1]

    dst_dir = 'dst_folder'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for cand in cands:
        
        dst_path = os.path.join(dst_dir, str(cand))
        if os.path.exists(dst_path):
            continue
      
        #print(cand, info[cand]['miou_with_dice'])
        os.system('cp -r {} \'{}\''.format('template', dst_path))
        with open(os.path.join(dst_path, 'arch.pkl'), 'wb') as f:
            pickle.dump(cand, f)
if __name__ == '__main__':
    main()
