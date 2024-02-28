
import copy

import torch
import torch.nn as nn

from model.image_translator.networks import FewShotGen, GPPatchMcResDis

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class Translator_Model(nn.Module):
    def __init__(self, hp):
        super(Translator_Model, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)
        
    def forward(self, co_data, cl_data, hp, mode):
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()
        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa)
            s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)  # translation
            xr = self.gen.decode(c_xa, s_xa)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb)
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la)
            _, xb_gan_feat = self.dis(xb, lb)
            _, xa_gan_feat = self.dis(xa, la)
            l_c_rec = recon_criterion(xr_gan_feat.mean(3).mean(2),
                                      xa_gan_feat.mean(3).mean(2))
            l_m_rec = recon_criterion(xt_gan_feat.mean(3).mean(2),
                                      xb_gan_feat.mean(3).mean(2))
            l_x_rec = recon_criterion(xr, xa)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
                'fm_w'] * (l_c_rec + l_m_rec))
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc
        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb, lb)
            l_real = hp['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach(),
                                                                  lb)
            l_fake = hp['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
        elif mode == 'picker_update':
            qry_features = self.gen.enc_content(xb).mean((2,3))
            nb_features = self.gen.enc_content(xa).mean((2,3))
            matrix_forward = torch.mm(qry_features, nb_features.transpose(1,0))
            matrix_reverse = torch.mm(nb_features, qry_features.transpose(0,1))
            scores_forward = self.get_score(qry = xb, nb = xa)
            scores_reverse = self.get_score(qry = xa, nb = xb)
            loss_forward = recon_criterion(matrix_forward, scores_forward)
            loss_reverse = recon_criterion(matrix_reverse, scores_reverse)
            loss = loss_forward + loss_reverse * 0.5
            loss *= 0.01
            loss.backward()
            return loss
        elif mode == 'traffic_picker_update':
            qry_features = self.gen.enc_content(xb).mean((2,3))
            nb_features = self.gen.enc_content(xa).mean((2,3))
            matrix_forward = torch.mm(qry_features, nb_features.transpose(1,0))
            matrix_reverse = torch.mm(nb_features, qry_features.transpose(0,1))
            scores_forward = self.get_traffic_score(qry = xb, nb = xa)
            scores_reverse = self.get_traffic_score(qry = xa, nb = xb)
            loss_forward = recon_criterion(matrix_forward, scores_forward)
            loss_reverse = recon_criterion(matrix_reverse, scores_reverse)
            loss = loss_forward + loss_reverse * 0.5
            loss *= 0.01
            loss.backward()
            return loss
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen.enc_content(xa)
        s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xa_current, s_xa_current)
        c_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)
        self.train()
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def get_score(self, qry, nb):
        with torch.no_grad():
            c_xa = self.gen.enc_content(nb.detach())
            s_xb = self.gen.enc_class_model(qry.detach())
            translation = self.gen.decode(c_xa, s_xb)
            fake_degree = self.dis.get_quality(qry, translation)
        return fake_degree
    
    def get_traffic_score(self, qry, nb):
        with torch.no_grad():
            s_xb = self.gen.enc_class_model(nb.detach())
            c_xa = self.gen.enc_content(qry.detach())
            translation = self.gen.decode(c_xa, s_xb)
            fake_degree = self.dis.get_quality(qry, translation)
        return fake_degree
 

    def pick_traffic(self, picker, qry, picker_loader, expansion_size=0, get_img=False, random=False, img_id='', get_original=True, type='image_translator'):
        if expansion_size == 0:
            get_original = True
        candidate_neighbours = next(iter(picker_loader)) # from train sampler, size: pool_size, 3, h, w + label_size
        candidate_neighbours = candidate_neighbours[0].cuda() # extracts img from img+label
        assert len(candidate_neighbours) >= expansion_size
        with torch.no_grad():
            qry_features = picker.enc_content(qry).mean((2,3)) # batch=1, feature_size
            nb_features = picker.enc_content(candidate_neighbours).mean((2,3))
            nb_features_trans = nb_features.transpose(1,0)
            scores = torch.mm(qry_features, nb_features_trans).squeeze() # q qries, n neighbors
        if random == False:
            scores, idxs = torch.sort(scores, descending=False) # best (lower KL divergence) in front
            idxs = idxs.long()
            selected_nbs = candidate_neighbours.index_select(dim=0, index=idxs)
            selected_nbs = selected_nbs[:expansion_size, :, :, :]
        else:
            selected_nbs = candidate_neighbours[:expansion_size, :, :, :]

        class_code = self.compute_k_style(qry, 1)
        translated_qry = self.translate_simple(qry, class_code)
        if get_original == True:
            translations = [translated_qry]
        else:
            translations = []
        
        for selected_i in range(expansion_size):
            nb = selected_nbs[selected_i, :, :, :].unsqueeze(0)
            if type == 'image_translator' or type == 'random-image_translator':
                class_code = self.compute_k_style(nb, 1)
                translation = self.translate_simple(qry, class_code)
            elif type == 'mix-up' or type == 'random-mix-up':
                nb = self.translate_simple(nb, self.compute_k_style(nb, 1))
                translation = 0.5 * (nb + translated_qry)
            translations.append(translation)

        if get_img == True:
            return translations
        else:
            return torch.stack(translations).squeeze()

    def pick_animals(self, picker, qry, picker_loader, expansion_size=0, get_img = False, random=False, img_id='', get_original=True, type='image_translator'): 
        if expansion_size == 0:
            get_original = True
        candidate_neighbours = next(iter(picker_loader)) # from train sampler, size: pool_size, 3, h, w + label_size
        candidate_neighbours = candidate_neighbours[0].cuda() # extracts img from img+label
        assert len(candidate_neighbours) >= expansion_size
        with torch.no_grad():
            qry_features = picker.enc_content(qry).mean((2,3)) # batch=1, feature_size
            nb_features = picker.enc_content(candidate_neighbours).mean((2,3))
            nb_features_trans = nb_features.transpose(1,0)
            scores = torch.mm(qry_features, nb_features_trans).squeeze() # q qries, n neighbors
        if random == False:
            scores, idxs = torch.sort(scores, descending=False) # best (lower KL divergence) in front
            idxs = idxs.long()
            selected_nbs = candidate_neighbours.index_select(dim=0, index=idxs)
            selected_nbs = selected_nbs[:expansion_size, :, :, :]
        else:
            selected_nbs = candidate_neighbours[:expansion_size, :, :, :]
        class_code = self.compute_k_style(qry, 1)
        qry = self.translate_simple(qry, class_code)
        if get_original == True:
            translations = [qry]
        else:
            translations = []
        
        
        for selected_i in range(expansion_size):
            nb = selected_nbs[selected_i, :, :, :].unsqueeze(0)
            if type == 'image_translator' or type == 'random-image_translator':
                translation = self.translate_simple(nb, class_code)
            elif type == 'mix-up' or type == 'random-mix-up':
                nb = self.translate_simple(nb, self.compute_k_style(nb, 1))
                translation = 0.5 * (nb + qry)
            translations.append(translation)

        if get_img == True:
            return translations
        else:
            return torch.stack(translations).squeeze()
