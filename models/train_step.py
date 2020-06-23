
    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        if(self.step == 0):
            self.optimizer.zero_grad()

        if(self.step % 2 == 0):
            self.optimizer.step()            
            self.optimizer.zero_grad()

        imgs = imgs.to(self.device)
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        epsilon = cal_epsilon(
            self.args.decay_k, self.total_step, self.args.sample_method)
        logits = self.model(imgs, tgt4training, epsilon)

        # calculate loss
        loss = cal_loss(logits, tgt4cal_loss)/2
        self.step += 1
        self.total_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        
        return loss.item()