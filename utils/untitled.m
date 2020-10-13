[v, ind] = sort(class);

figure
plot(v, 'g.')
hold on
plot(pdfpred(ind), 'r.')
plot(cdfpred(ind), 'b.')
legend('GT-age', 'pdf-age', 'cdf-age')
title(sprintf('pdf-mae:%.3f, cdf-mae:%.3f\n', mean(abs(pdfpred-class)),  mean(abs(cdfpred-class)) ))


id = find(class>50)
mae = mean(abs(pdfpred(id)-class(id)))

mae = mean(abs(pdfpred(id)-class(id)))

[v, ind] = sort(class);
pred = (pred1+pred2)./2;


[v, ind] = sort(class);
pred = (exp1+exp2)./2;
figure
plot(v, 'g.')
hold on
plot(pdfpred(ind), 'r.')
legend('GT-age', 'pdf-age', 'cdf-age')
title(sprintf('mae:%.3f\n', mean(abs(pred-class))))


mean(abs(pred-class))
id = find(class>50)
mae = mean(abs(pred(id)-class(id)))

plot(pred(ind) -v)


pred1 - pred2