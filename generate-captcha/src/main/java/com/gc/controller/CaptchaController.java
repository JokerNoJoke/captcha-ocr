package com.gc.controller;

import java.io.IOException;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import com.wf.captcha.SpecCaptcha;
import com.wf.captcha.base.Captcha;

import jakarta.servlet.http.HttpServletResponse;

@RestController
public class CaptchaController {

    @GetMapping("captcha")
    public void captcha(HttpServletResponse response, String uuid) throws IOException {
        response.setContentType("image/gif");
        response.setHeader("Pragma", "No-cache");
        response.setHeader("Cache-Control", "no-cache");
        response.setDateHeader("Expires", 0);

        // 生成验证码
        SpecCaptcha captcha = new SpecCaptcha(150, 40);
        response.setHeader("CAPTCHA", captcha.text());
        captcha.setLen(5);
        captcha.setCharType(Captcha.TYPE_DEFAULT);
        captcha.out(response.getOutputStream());
    }

}
